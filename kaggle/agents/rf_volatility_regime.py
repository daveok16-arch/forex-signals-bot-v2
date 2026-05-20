#!/usr/bin/env python3
"""
Agent C: Random Forest Volatility Regime Model
Trains on volatility features to predict high-conviction directional moves.
Output: volatility_regime_rf.onnx
"""

# Install ONNX converter if missing
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
# CONFIG
# ============================================================
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD", "BTCUSD"]
HORIZON = 2  # 2-hour ahead prediction (tighter, volatility-based)

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
# FEATURE ENGINEERING (Volatility-centric)
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

        # Rolling volatility features
        pdf["returns"] = pdf["close"].pct_change().fillna(0)
        pdf["vol_14"] = pdf["returns"].rolling(14).std()
        pdf["vol_28"] = pdf["returns"].rolling(28).std()
        pdf["vol_ratio"] = pdf["vol_14"] / (pdf["vol_28"] + 1e-9)

        # Bollinger bandwidth
        sma_20 = pdf["close"].rolling(20).mean()
        std_20 = pdf["close"].rolling(20).std()
        pdf["bb_width"] = (2 * std_20) / (sma_20 + 1e-9)

        # ATR relative to price
        pdf["atr_rel"] = pdf["atr_14"] / pdf["close"]

        # Price position within daily range (0-1)
        pdf["daily_high"] = pdf["high"].rolling(24).max()
        pdf["daily_low"] = pdf["low"].rolling(24).min()
        pdf["range_position"] = (pdf["close"] - pdf["daily_low"]) / (pdf["daily_high"] - pdf["daily_low"] + 1e-9)

        # Momentum
        pdf["mom_4"] = pdf["close"].pct_change(4)
        pdf["mom_12"] = pdf["close"].pct_change(12)

        # Skewness of returns (asymmetry)
        pdf["skew_14"] = pdf["returns"].rolling(14).skew()

        # Target: strong directional move up in next 2 bars (>0.15% for FX, >0.5% for XAU, >1% for BTC)
        thresholds = {"XAUUSD": 0.005, "BTCUSD": 0.01}
        thresh = thresholds.get(pair, 0.0015)

        future_return = pdf["close"].shift(-HORIZON).pct_change(HORIZON)
        pdf["target"] = (future_return > thresh).astype(float)

        # Select features
        feat_cols = [
            "rsi_14", "ema_20_50_cross", "bollinger_position",
            "vol_14", "vol_28", "vol_ratio", "bb_width", "atr_rel",
            "range_position", "mom_4", "mom_12", "skew_14", "macd_hist"
        ]

        pdf = pdf[feat_cols + ["target"]].dropna()
        pdf["pair"] = pair
        all_features.append(pdf)

    combined = pd.concat(all_features, ignore_index=True)
    return combined, feat_cols

# ============================================================
# MODEL TRAINING
# ============================================================
def train_rf(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score((y_val > 0.5).astype(int), (val_pred > 0.5).astype(int))
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f"      Val Accuracy: {val_acc:.2%} | RMSE: {val_rmse:.4f}")

    # Feature importance
    importances = dict(zip(model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"f{i}" for i in range(len(X_train[0]))], 
                          model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"      Top features: {top}")

    return model

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, feature_names, path="/kaggle/working/volatility_regime_rf.onnx"):
    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=14)

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
    print("AGENT C: RANDOM FOREST VOLATILITY REGIME")
    print("=" * 60)

    print("[1/4] Loading ETL data...")
    df = load_data()

    print("[2/4] Engineering volatility features...")
    combined, feat_cols = engineer_volatility_features(df)
    print(f"      Samples: {len(combined)} | Features: {feat_cols}")

    X = combined[feat_cols].values.astype(np.float32)
    y = combined["target"].values.astype(np.float32)

    if len(X) < 1000:
        raise RuntimeError("Insufficient volatility data.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y>0.5).astype(int)
    )

    print("[3/4] Training Random Forest...")
    model = train_rf(X_train, y_train, X_val, y_val)

    print("[4/4] Exporting ONNX...")
    export_onnx(model, feat_cols)

    # Save metadata
    meta = {
        "agent": "volatility_regime_rf",
        "feature_cols": feat_cols,
        "samples": len(X),
        "pos_rate": float(y.mean()),
    }
    import json
    with open("/kaggle/working/volatility_regime_rf_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("AGENT C COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
