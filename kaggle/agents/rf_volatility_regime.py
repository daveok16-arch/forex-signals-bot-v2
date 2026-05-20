#!/usr/bin/env python3
"""
Agent C: Random Forest Volatility Regime Model v2.2
Trains on volatility features to predict high-conviction directional moves.
Output: volatility_regime_rf.onnx

PATCH: Robust dataset path discovery via inline data_loader.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "skl2onnx", "--quiet"])

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from skl2onnx import convert_sklearn
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

def validate_dataset(df, min_rows=1000):
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
HORIZON = 2

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_volatility_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["pair", "datetime"]).reset_index(drop=True)
    all_features = []
    for pair in df["pair"].unique():
        pdf = df[df["pair"] == pair].copy()
        if len(pdf) < 50:
            continue
        pdf["returns"] = pdf["close"].pct_change().fillna(0)
        pdf["vol_14"] = pdf["returns"].rolling(14).std()
        pdf["vol_28"] = pdf["returns"].rolling(28).std()
        pdf["vol_ratio"] = pdf["vol_14"] / (pdf["vol_28"] + 1e-9)
        sma_20 = pdf["close"].rolling(20).mean()
        std_20 = pdf["close"].rolling(20).std()
        pdf["bb_width"] = (2 * std_20) / (sma_20 + 1e-9)
        pdf["atr_rel"] = pdf["atr_14"] / pdf["close"]
        pdf["daily_high"] = pdf["high"].rolling(24).max()
        pdf["daily_low"] = pdf["low"].rolling(24).min()
        pdf["range_position"] = (pdf["close"] - pdf["daily_low"]) / (pdf["daily_high"] - pdf["daily_low"] + 1e-9)
        pdf["mom_4"] = pdf["close"].pct_change(4)
        pdf["mom_12"] = pdf["close"].pct_change(12)
        pdf["skew_14"] = pdf["returns"].rolling(14).skew()
        thresholds = {"XAUUSD": 0.005, "BTCUSD": 0.01}
        thresh = thresholds.get(pair, 0.0015)
        future_return = pdf["close"].shift(-HORIZON).pct_change(HORIZON)
        pdf["target"] = (future_return > thresh).astype(float)
        feat_cols = ["rsi_14", "ema_20_50_cross", "bollinger_position", "vol_14", "vol_28", "vol_ratio",
                     "bb_width", "atr_rel", "range_position", "mom_4", "mom_12", "skew_14", "macd_hist"]
        pdf = pdf[feat_cols + ["target"]].dropna()
        pdf["pair"] = pair
        all_features.append(pdf)
    combined = pd.concat(all_features, ignore_index=True)
    return combined, feat_cols

# ============================================================
# TRAINING
# ============================================================
def train_rf(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_acc = accuracy_score((y_val > 0.5).astype(int), (val_pred > 0.5).astype(int))
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f"Val Accuracy: {val_acc:.2%} | RMSE: {val_rmse:.4f}")
    return model

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, feature_names, path="/kaggle/working/volatility_regime_rf.onnx"):
    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=14)
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
    print("AGENT C: RANDOM FOREST VOLATILITY REGIME v2.2")
    print("=" * 60)

    print("[1/4] Loading ETL data...")
    df = load_forex_data()

    print("[2/4] Validating dataset...")
    validate_dataset(df)

    print("[3/4] Engineering volatility features...")
    combined, feat_cols = engineer_volatility_features(df)
    print(f"Samples: {len(combined)} | Features: {feat_cols}")
    X = combined[feat_cols].values.astype(np.float32)
    y = combined["target"].values.astype(np.float32)
    if len(X) < 1000:
        raise RuntimeError("Insufficient volatility data.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y>0.5).astype(int))

    print("[4/4] Training Random Forest...")
    model = train_rf(X_train, y_train, X_val, y_val)
    export_onnx(model, feat_cols)

    meta = {"agent": "volatility_regime_rf", "feature_cols": feat_cols, "samples": len(X), "pos_rate": float(y.mean())}
    with open("/kaggle/working/volatility_regime_rf_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("AGENT C COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
