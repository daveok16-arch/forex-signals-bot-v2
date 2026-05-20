#!/usr/bin/env python3
"""
Agent B: XGBoost Macro-Sentiment Model
Trains on daily macro features + pair aggregated stats.
Output: macro_sentiment_xgb.onnx
"""

# Install ONNX converter if missing
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
# CONFIG
# ============================================================
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD", "BTCUSD"]
HORIZON = 4  # 4-hour directional prediction using daily macro

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    path = "/kaggle/input/forex-raw-data/forex_features.parquet"
    if not os.path.exists(path):
        path = "/kaggle/input/forex-raw-data/forex_features.csv"
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_macro_features(df):
    df = df.copy()

    # Ensure datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date

    # We use 1h data but aggregate macro per day
    # For each pair, compute daily aggregates
    daily = []

    for pair in df["pair"].unique():
        pdf = df[df["pair"] == pair].copy()
        pdf = pdf.sort_values("datetime")

        # Daily aggregation
        day = pdf.groupby("date").agg({
            "close": ["first", "last", "min", "max"],
            "atr_14": "mean",
            "rsi_14": "mean",
            "dxy_index": "last",
            "vix_proxy": "last",
            "yield_spread_us_de": "last",
        }).reset_index()

        # Flatten multi-index columns
        day.columns = ["date", "open", "close", "low", "high", "avg_atr", "avg_rsi", "dxy", "vix", "yield_spread"]
        day["pair"] = pair
        day["daily_return"] = (day["close"] - day["open"]) / day["open"]
        day["daily_range"] = (day["high"] - day["low"]) / day["open"]

        # Forward-fill macro missing values
        for col in ["dxy", "vix", "yield_spread"]:
            day[col] = day[col].ffill().bfill()

        # Lag features (macro moves slowly)
        for col in ["dxy", "vix", "yield_spread", "avg_rsi"]:
            day[f"{col}_chg_1d"] = day[col].diff(1)
            day[f"{col}_chg_3d"] = day[col].diff(3)

        # Target: next day direction (1 if close > open next day, else 0)
        day["target"] = (day["daily_return"].shift(-1) > 0.001).astype(float)

        daily.append(day)

    daily = pd.concat(daily, ignore_index=True)
    daily = daily.dropna()
    return daily

# ============================================================
# MODEL TRAINING
# ============================================================
def train_xgb(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="rmse"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )

    # Evaluate
    val_pred = model.predict(X_val)
    # Convert to binary for accuracy
    val_acc = accuracy_score((y_val > 0.5).astype(int), (val_pred > 0.5).astype(int))
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"      Val Accuracy: {val_acc:.2%} | AUC: {val_auc:.4f}")

    return model

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, feature_names, path="/kaggle/working/macro_sentiment_xgb.onnx"):
    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=14)

    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ ONNX exported: {path}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    dummy = np.random.randn(1, len(feature_names)).astype(np.float32)
    out = sess.run(None, {"input": dummy})[0]
    print(f"   Verification output shape: {out.shape} | Sample: {out[0]:.4f}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("AGENT B: XGBOOST MACRO-SENTIMENT")
    print("=" * 60)

    print("[1/4] Loading ETL data...")
    df = load_data()

    print("[2/4] Engineering macro features...")
    daily = engineer_macro_features(df)
    print(f"      Daily samples: {len(daily)}")

    feature_cols = [c for c in daily.columns if c not in ["date", "pair", "target", "open", "close", "low", "high"]]
    print(f"      Features: {feature_cols}")

    X = daily[feature_cols].values.astype(np.float32)
    y = daily["target"].values.astype(np.float32)

    if len(X) < 200:
        raise RuntimeError("Insufficient macro data. Need more historical ETL runs.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=(y>0.5).astype(int))

    print("[3/4] Training XGBoost...")
    model = train_xgb(X_train, y_train, X_val, y_val)

    print("[4/4] Exporting ONNX...")
    export_onnx(model, feature_cols)

    # Save metadata
    meta = {
        "agent": "macro_sentiment_xgb",
        "feature_cols": feature_cols,
        "samples": len(X),
        "pos_rate": float(y.mean()),
    }
    import json
    with open("/kaggle/working/macro_sentiment_xgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("AGENT B COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
