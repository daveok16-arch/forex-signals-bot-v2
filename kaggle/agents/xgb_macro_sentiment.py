#!/usr/bin/env python3
"""
Agent B: XGBoost Macro-Sentiment Model v2.2
Trains on daily macro features + pair aggregated stats.
Output: macro_sentiment_xgb.onnx

PATCH: Robust dataset path discovery via inline data_loader.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxmltools", "--quiet"])

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType

# ============================================================
# INLINED DATA LOADER v2.2
# ============================================================
POSSIBLE_PATHS = [
    "/kaggle/input/datasets/chamberbot/forex-raw-data/forex_features.parquet",
    "/kaggle/input/datasets/chamberbot/forex-raw-data/forex_features.csv",
    "/kaggle/input/forex-raw-data/forex_features.parquet",
    "/kaggle/input/forex-raw-data/forex_features.csv",
    "/kaggle/input/chamberbot-forex-raw-data/forex_features.parquet",
    "/kaggle/input/chamberbot-forex-raw-data/forex_features.csv",
    "./forex_features.parquet", "./forex_features.csv",
    "../forex_features.parquet", "../forex_features.csv",
    "/kaggle/working/forex_etl_output/forex_features.parquet",
    "/kaggle/working/forex_etl_output/forex_features.csv",
]

def load_forex_data(verbose=True):
    paths_to_check = list(POSSIBLE_PATHS)
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for fname in files:
                if fname in ("forex_features.parquet", "forex_features.csv"):
                    full = os.path.join(root, fname)
                    if full not in paths_to_check:
                        paths_to_check.append(full)
    found_path = None
    checked = []
    for p in paths_to_check:
        checked.append(p)
        if os.path.exists(p):
            found_path = p
            break
    if verbose:
        print(f"[data_loader v2.2] Dataset discovery:")
        for p in checked:
            status = "FOUND" if p == found_path else "missing"
            print(f"    {status}: {p}")
    if found_path is None:
        debug = []
        if os.path.exists("/kaggle/input"):
            debug.append("Contents of /kaggle/input:")
            for item in os.listdir("/kaggle/input"):
                debug.append(f"  - {item}")
                sub = os.path.join("/kaggle/input", item)
                if os.path.isdir(sub):
                    try:
                        for s in os.listdir(sub):
                            debug.append(f"      - {s}")
                    except:
                        pass
        raise FileNotFoundError(f"Dataset not found. Checked {len(checked)} paths.\n" + "\n".join(debug))
    if verbose:
        print(f"[data_loader] Loading from: {found_path}")
    if found_path.endswith(".parquet"):
        df = pd.read_parquet(found_path)
    else:
        df = pd.read_csv(found_path)
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    required = ["close", "pair", "timeframe"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Have: {list(df.columns)}")
    if verbose:
        print(f"[data_loader] Shape: {df.shape} | Pairs: {df['pair'].nunique()} | TFs: {df['timeframe'].nunique()}")
    return df

def validate_dataset(df, min_rows=200):
    issues = []
    if len(df) < min_rows:
        issues.append(f"Only {len(df)} rows (min: {min_rows})")
    if df["pair"].nunique() < 2:
        issues.append(f"Only {df['pair'].nunique()} pairs")
    nan_cols = [c for c in df.columns if df[c].isna().all()]
    if nan_cols:
        issues.append(f"All-NaN columns: {nan_cols}")
    if issues:
        raise RuntimeError("Validation failed:\n" + "\n".join(f"  - {i}" for i in issues))
    print(f"[data_loader] Validation passed: {len(df)} rows")

# ============================================================
# CONFIG
# ============================================================
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD", "BTCUSD"]
HORIZON = 4

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_macro_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    daily = []
    for pair in df["pair"].unique():
        pdf = df[df["pair"] == pair].copy()
        pdf = pdf.sort_values("datetime")
        day = pdf.groupby("date").agg({
            "close": ["first", "last", "min", "max"],
            "atr_14": "mean",
            "rsi_14": "mean",
            "dxy_index": "last",
            "vix_proxy": "last",
            "yield_spread_us_de": "last",
        }).reset_index()
        day.columns = ["date", "open", "close", "low", "high", "avg_atr", "avg_rsi", "dxy", "vix", "yield_spread"]
        day["pair"] = pair
        day["daily_return"] = (day["close"] - day["open"]) / day["open"]
        day["daily_range"] = (day["high"] - day["low"]) / day["open"]
        for col in ["dxy", "vix", "yield_spread"]:
            day[col] = day[col].ffill().bfill()
        for col in ["dxy", "vix", "yield_spread", "avg_rsi"]:
            day[f"{col}_chg_1d"] = day[col].diff(1)
            day[f"{col}_chg_3d"] = day[col].diff(3)
        day["target"] = (day["daily_return"].shift(-1) > 0.001).astype(float)
        daily.append(day)
    daily = pd.concat(daily, ignore_index=True)
    daily = daily.dropna()
    return daily

# ============================================================
# TRAINING
# ============================================================
def train_xgb(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
        objective="reg:squarederror", eval_metric="rmse"
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    val_pred = model.predict(X_val)
    val_acc = accuracy_score((y_val > 0.5).astype(int), (val_pred > 0.5).astype(int))
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"Val Accuracy: {val_acc:.2%} | AUC: {val_auc:.4f}")
    return model

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, feature_names, path="/kaggle/working/macro_sentiment_xgb.onnx"):
    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=14)
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX exported: {path}")
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    dummy = np.random.randn(1, len(feature_names)).astype(np.float32)
    out = sess.run(None, {"input": dummy})[0]
    print(f"Verification: shape={out.shape} sample={out[0]:.4f}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("AGENT B: XGBOOST MACRO-SENTIMENT v2.2")
    print("=" * 60)

    print("[1/4] Loading ETL data...")
    df = load_forex_data()

    print("[2/4] Validating dataset...")
    validate_dataset(df)

    print("[3/4] Engineering macro features...")
    daily = engineer_macro_features(df)
    print(f"Daily samples: {len(daily)}")
    feature_cols = [c for c in daily.columns if c not in ["date", "pair", "target", "open", "close", "low", "high"]]
    print(f"Features: {feature_cols}")
    X = daily[feature_cols].values.astype(np.float32)
    y = daily["target"].values.astype(np.float32)
    if len(X) < 200:
        raise RuntimeError("Insufficient macro data.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y>0.5).astype(int))

    print("[4/4] Training XGBoost...")
    model = train_xgb(X_train, y_train, X_val, y_val)
    export_onnx(model, feature_cols)

    meta = {"agent": "macro_sentiment_xgb", "feature_cols": feature_cols, "samples": len(X), "pos_rate": float(y.mean())}
    with open("/kaggle/working/macro_sentiment_xgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("AGENT B COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
