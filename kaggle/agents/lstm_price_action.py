#!/usr/bin/env python3
"""
Agent A: LSTM Price Action Model
Trains on 1h technical sequences to predict next directional move.
Output: price_action_lstm.onnx
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================
SEQ_LEN = 48          # 48 hours of history
PRED_HORIZON = 4      # Predict direction 4 hours ahead
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "rsi_14", "ema_20_50_cross", "bollinger_position",
    "macd_hist", "atr_14", "returns"
]

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD", "BTCUSD"]

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    path = "/kaggle/input/forex-raw-data/forex_features.parquet"
    if not os.path.exists(path):
        path = "/kaggle/input/forex-raw-data/forex_features.csv"
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

    # Ensure required columns exist
    for c in FEATURE_COLS + ["close", "pair", "timeframe"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Filter 1h timeframe
    df = df[df["timeframe"] == "1h"].copy()
    df = df.sort_values(["pair", "datetime"]).reset_index(drop=True)
    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    df = df.copy()

    # Normalize RSI to 0-1
    df["rsi_14"] = df["rsi_14"] / 100.0

    # Normalize ATR by recent price (approximate using Close)
    df["atr_14_norm"] = df["atr_14"] / df["close"]

    # Normalize MACD hist by Close
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]

    # Compute returns if not present, else normalize
    if "returns" not in df.columns or df["returns"].isna().all():
        df["returns"] = df["close"].pct_change()
    df["returns"] = df["returns"].fillna(0)

    # Clip extreme outliers
    for col in ["rsi_14", "bollinger_position", "macd_hist_norm", "atr_14_norm", "returns"]:
        df[col] = df[col].clip(-5, 5)

    feature_cols = [
        "rsi_14", "ema_20_50_cross", "bollinger_position",
        "macd_hist_norm", "atr_14_norm", "returns"
    ]
    return df, feature_cols

# ============================================================
# SEQUENCE CREATION
# ============================================================
def create_sequences(df, feature_cols, seq_len=48, horizon=4):
    X, y = [], []

    for pair in df["pair"].unique():
        pdf = df[df["pair"] == pair].reset_index(drop=True)
        if len(pdf) < seq_len + horizon + 10:
            continue

        scaler = RobustScaler()
        pdf[feature_cols] = scaler.fit_transform(pdf[feature_cols])

        vals = pdf[feature_cols].values.astype(np.float32)
        closes = pdf["close"].values

        for i in range(seq_len, len(vals) - horizon):
            seq = vals[i - seq_len:i]
            future_return = (closes[i + horizon] - closes[i]) / closes[i]

            # Label: 1.0 if up > 0.1%, 0.0 if down > 0.1%, else skip (reduce noise)
            if future_return > 0.001:
                label = 1.0
            elif future_return < -0.001:
                label = 0.0
            else:
                continue

            X.append(seq)
            y.append(label)

    return np.array(X), np.array(y)

# ============================================================
# PYTORCH DATASET
# ============================================================
class ForexDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class LSTMPriceAction(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last = lstm_out[:, -1, :]
        x = self.fc1(last)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# ============================================================
# TRAINING
# ============================================================
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, input_dim, seq_len, path="/kaggle/working/price_action_lstm.onnx"):
    model.eval()
    dummy = torch.randn(1, seq_len, input_dim).to(DEVICE)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["probability"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "probability": {0: "batch_size"}
        },
        opset_version=14
    )
    print(f"✅ ONNX exported: {path}")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    test_out = sess.run(None, {"input": dummy.cpu().numpy()})[0]
    print(f"   Verification output shape: {test_out.shape} | Sample: {test_out[0,0]:.4f}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("AGENT A: LSTM PRICE ACTION")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("[1/5] Loading ETL data...")
    df = load_data()
    print(f"      Rows: {len(df)} | Pairs: {df['pair'].nunique()}")

    print("[2/5] Engineering features...")
    df, feature_cols = engineer_features(df)
    print(f"      Features: {feature_cols}")

    print("[3/5] Creating sequences...")
    X, y = create_sequences(df, feature_cols, SEQ_LEN, PRED_HORIZON)
    print(f"      Sequences: {len(X)} | Positives: {y.mean():.2%}")

    if len(X) < 1000:
        raise RuntimeError("Insufficient training data. Check ETL dataset.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_ds = ForexDataset(X_train, y_train)
    val_ds = ForexDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("[4/5] Training LSTM...")
    model = LSTMPriceAction(len(feature_cols))
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    # Validation accuracy
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            all_preds.extend(pred.flatten())
            all_true.extend(yb.numpy().flatten())

    acc = np.mean((np.array(all_preds) > 0.5) == (np.array(all_true) > 0.5))
    print(f"      Validation Accuracy: {acc:.2%}")

    print("[5/5] Exporting ONNX...")
    export_onnx(model, len(feature_cols), SEQ_LEN)

    # Save metadata
    meta = {
        "agent": "price_action_lstm",
        "val_accuracy": float(acc),
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "horizon": PRED_HORIZON,
        "device": str(DEVICE),
        "samples": len(X)
    }
    import json
    with open("/kaggle/working/price_action_lstm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("AGENT A COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
