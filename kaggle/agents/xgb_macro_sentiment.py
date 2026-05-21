#!/usr/bin/env python3
"""
Agent B: XGBoost Macro-Sentiment Model v2.3
Trains on daily macro features + pair aggregated stats.
Output: macro_sentiment_xgb.onnx

PATCH v2.3: Full fault tolerance for missing macro columns.
- Dynamically builds agg_dict from existing columns only
- Gracefully handles missing DXY, VIX, or yield spreads
- Model trains with whatever macro data is available
- Compatible with v2.2 unified data_loader
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
# INLINED DATA LOADER v2.2 (unchanged from v2.2)
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
# FEATURE ENGINEERING v2.3 (FAULT TOLERANT)
# ============================================================
def engineer_macro_features(df):
    """
    Build daily macro features with dynamic column detection.
    Survives missing DXY, VIX, or yield spreads.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date

    # Define which columns we WANT for aggregation
    desired_agg = {
        "close": ["first", "last", "min", "max"],
        "atr_14": "mean",
        "rsi_14": "mean",
    }

    # Define which macro columns we WANT (may or may not exist)
    desired_macro = ["dxy_index", "vix_proxy", "yield_spread_us_de"]

    # Build actual agg dict: only include columns that exist in the dataframe
    actual_agg = {}
    for col, agg_func in desired_agg.items():
        if col in df.columns:
            actual_agg[col] = agg_func

    available_macro = [m for m in desired_macro if m in df.columns]
    for m in available_macro:
        actual_agg[m] = "last"

    print(f"[macro_engineer] Available macro columns: {available_macro}")
    print(f"[macro_engineer] Aggregation columns: {list(actual_agg.keys())}")

    daily = []

    for pair in df["pair"].unique():
        pdf = df[df["pair"] == pair].copy()
        pdf = pdf.sort_values("datetime")

        # Aggregate only existing columns
        day = pdf.groupby("date").agg(actual_agg).reset_index()

        # Flatten multi-index columns if any
        if isinstance(day.columns, pd.MultiIndex):
            day.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in day.columns.values]

        # Standardize column names after aggregation
        # close_first -> open, close_last -> close, close_min -> low, close_max -> high
        rename_map = {}
        for c in day.columns:
            if c == "close_first" or c == "close_first_":
                rename_map[c] = "open"
            elif c == "close_last" or c == "close_last_":
                rename_map[c] = "close"
            elif c == "close_min" or c == "close_min_":
                rename_map[c] = "low"
            elif c == "close_max" or c == "close_max_":
                rename_map[c] = "high"
            elif c == "atr_14_mean" or c == "atr_14_mean_":
                rename_map[c] = "avg_atr"
            elif c == "rsi_14_mean" or c == "rsi_14_mean_":
                rename_map[c] = "avg_rsi"
            elif c in ["dxy_index_last", "dxy_index_last_"]:
                rename_map[c] = "dxy"
            elif c in ["vix_proxy_last", "vix_proxy_last_"]:
                rename_map[c] = "vix"
            elif c in ["yield_spread_us_de_last", "yield_spread_us_de_last_"]:
                rename_map[c] = "yield_spread"

        if rename_map:
            day = day.rename(columns=rename_map)

        day["pair"] = pair

        # Compute derived features only if base columns exist
        if "open" in day.columns and "close" in day.columns:
            day["daily_return"] = (day["close"] - day["open"]) / day["open"]
        else:
            day["daily_return"] = 0.0

        if "low" in day.columns and "high" in day.columns and "open" in day.columns:
            day["daily_range"] = (day["high"] - day["low"]) / day["open"]
        else:
            day["daily_range"] = 0.0

        # Forward-fill macro columns that exist
        macro_cols = ["dxy", "vix", "yield_spread"]
        for col in macro_cols:
            if col in day.columns:
                day[col] = day[col].ffill().bfill()

        # Lag features: only for macro columns that exist
        lag_candidates = ["dxy", "vix", "yield_spread", "avg_rsi"]
        for col in lag_candidates:
            if col in day.columns:
                day[f"{col}_chg_1d"] = day[col].diff(1)
                day[f"{col}_chg_3d"] = day[col].diff(3)

        # Target: next day direction
        if "daily_return" in day.columns:
            day["target"] = (day["daily_return"].shift(-1) > 0.001).astype(float)
        else:
            day["target"] = 0.5

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
    print("AGENT B: XGBOOST MACRO-SENTIMENT v2.3")
    print("=" * 60)

    print("[1/4] Loading ETL data...")
    df = load_forex_data()

    print("[2/4] Validating dataset...")
    validate_dataset(df)

    print("[3/4] Engineering macro features (fault tolerant)...")
    daily = engineer_macro_features(df)
    print(f"Daily samples: {len(daily)}")

    # Dynamically select feature columns (exclude non-feature columns)
    exclude = {"date", "pair", "target", "open", "close", "low", "high", "daily_return", "daily_range"}
    feature_cols = [c for c in daily.columns if c not in exclude]
    print(f"Dynamic features: {feature_cols}")

    if len(feature_cols) < 3:
        print("[WARN] Very few macro features available. Model may be weak.")

    X = daily[feature_cols].values.astype(np.float32)
    y = daily["target"].values.astype(np.float32)

    if len(X) < 200:
        raise RuntimeError("Insufficient macro data.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y>0.5).astype(int)
    )

    print("[4/4] Training XGBoost...")
    model = train_xgb(X_train, y_train, X_val, y_val)
    export_onnx(model, feature_cols)

    meta = {
        "agent": "macro_sentiment_xgb",
        "feature_cols": feature_cols,
        "samples": len(X),
        "pos_rate": float(y.mean()),
        "version": "2.3"
    }
    with open("/kaggle/working/macro_sentiment_xgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("AGENT B COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
