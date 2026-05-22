#!/usr/bin/env python3
# Agent A: LSTM Price Action Model v2.3.5
# Output: price_action_lstm.onnx
# Fixes: NaN/inf cleanup, CUDA isolation, no docstring corruption

import os
import json
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
# SAFE DEVICE DETECTION
# ============================================================
def get_safe_device():
    if not torch.cuda.is_available():
        print("[device] CUDA not available. Using CPU.")
        return torch.device("cpu")
    try:
        with torch.no_grad():
            a = torch.ones(3, 3, device="cuda")
            b = torch.ones(3, 3, device="cuda") * 2.0
            c = a + b
            torch.cuda.synchronize()
            del a, b, c
            torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"[device] CUDA OK: {device_name} (capability {capability[0]}.{capability[1]})")
        print(f"[device] PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
        return torch.device("cuda")
    except Exception as e:
        print(f"[device] CUDA test failed: {e}")
        print("[device] Falling back to CPU.")
        return torch.device("cpu")

DEVICE = get_safe_device()

# ============================================================
# DATA LOADER
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
        print("[data_loader] Dataset discovery:")
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
SEQ_LEN = 48
PRED_HORIZON = 4
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

FEATURE_COLS = ["rsi_14", "ema_20_50_cross", "bollinger_position", "macd_hist", "atr_14", "returns"]
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD", "BTCUSD"]

# ============================================================
# FEATURE ENGINEERING (NaN/inf cleanup)
# ============================================================
def engineer_features(df):
    df = df.copy()
    df["rsi_14"] = df["rsi_14"] / 100.0
    df["atr_14_norm"] = df["atr_14"] / df["close"]
    df["macd_hist_norm"] = df["macd_hist"] / df["close"]
    if "returns" not in df.columns or df["returns"].isna().all():
        df["returns"] = df["close"].pct_change()
    df["returns"] = df["returns"].fillna(0)
    for col in ["rsi_14", "bollinger_position", "macd_hist_norm", "atr_14_norm", "returns"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(0)
        df[col] = df[col].clip(-5, 5)
    feature_cols = ["rsi_14", "ema_20_50_cross", "bollinger_position", "macd_hist_norm", "atr_14_norm", "returns"]
    for col in feature_cols:
        if df[col].isna().any():
            print(f"[WARN] NaN found in {col} after cleanup — filling with 0")
            df[col] = df[col].fillna(0)
        if np.isinf(df[col]).any():
            print(f"[WARN] inf found in {col} after cleanup — replacing with 0")
            df[col] = df[col].replace([np.inf, -np.inf], 0)
    print("[features] NaN check passed. All columns clean.")
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
        if pdf[feature_cols].isna().any().any():
            print(f"[WARN] {pair} has NaN before scaling — filling")
            pdf[feature_cols] = pdf[feature_cols].fillna(0)
        scaler = RobustScaler()
        pdf[feature_cols] = scaler.fit_transform(pdf[feature_cols])
        pdf[feature_cols] = pdf[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        vals = pdf[feature_cols].values.astype(np.float32)
        closes = pdf["close"].values
        for i in range(seq_len, len(vals) - horizon):
            seq = vals[i - seq_len:i]
            future_return = (closes[i + horizon] - closes[i]) / closes[i]
            if future_return > 0.001:
                label = 1.0
            elif future_return < -0.001:
                label = 0.0
            else:
                continue
            X.append(seq)
            y.append(label)
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    if np.isnan(X_arr).any():
        print("[WARN] NaN in X array — replacing with 0")
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(y_arr).any():
        print("[WARN] NaN in y array — replacing with 0")
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return X_arr, y_arr

# ============================================================
# PYTORCH DATASET
# ============================================================
class ForexDataset(Dataset):
    def __init__(self, X, y):
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.tensor(X_clean, dtype=torch.float32)
        self.y = torch.tensor(y_clean, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# MODEL
# ============================================================
class LSTMPriceAction(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.0, bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        last = torch.nan_to_num(last, nan=0.0, posinf=5.0, neginf=-5.0)
        x = self.fc1(last)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.clamp(x, min=-10.0, max=10.0)
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
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"[FATAL] Model output has NaN/inf at epoch {epoch+1}")
                raise RuntimeError("Model output invalid")
            loss = criterion(pred, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[FATAL] Loss is NaN/inf at epoch {epoch+1}")
                raise RuntimeError("Loss became NaN")
            loss.backward()
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    print(f"[WARN] NaN gradient in {name} at epoch {epoch+1}")
            if has_nan_grad:
                optimizer.zero_grad()
                continue
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    return model

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, input_dim, seq_len, path="/kaggle/working/price_action_lstm.onnx"):
    model.eval()
    model_cpu = model.to("cpu")
    dummy = torch.randn(1, seq_len, input_dim)
    torch.onnx.export(
        model_cpu, dummy, path,
        input_names=["input"], output_names=["probability"],
        dynamic_axes={"input": {0: "batch_size"}, "probability": {0: "batch_size"}},
        opset_version=18
    )
    print(f"[onnx] Exported: {path}")
    # Optional verification — skip if onnxruntime not installed
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path)
        test_out = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"[onnx] Verification: shape={test_out.shape} | sample={test_out[0,0]:.4f}")
    except ImportError:
        print("[onnx] onnxruntime not installed — skipping verification")
    if DEVICE.type == "cuda":
        model.to(DEVICE)

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("AGENT A: LSTM PRICE ACTION v2.3.6")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    print("[1/5] Loading ETL data...")
    df = load_forex_data()
    print(f"      Rows: {len(df)} | Pairs: {df['pair'].nunique()}")
    print("[2/5] Validating dataset...")
    validate_dataset(df)
    print("[3/5] Engineering features...")
    df, feature_cols = engineer_features(df)
    print(f"      Features: {feature_cols}")
    print("[4/5] Creating sequences...")
    X, y = create_sequences(df, feature_cols, SEQ_LEN, PRED_HORIZON)
    print(f"      Sequences: {len(X)} | Positives: {y.mean():.2%}")
    if len(X) < 1000:
        raise RuntimeError("Insufficient training data.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_ds = ForexDataset(X_train, y_train)
    val_ds = ForexDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print("[5/5] Training LSTM...")
    model = LSTMPriceAction(len(feature_cols))
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
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
    print("[onnx] Exporting ONNX...")
    export_onnx(model, len(feature_cols), SEQ_LEN)
    meta = {
        "agent": "price_action_lstm",
        "val_accuracy": float(acc),
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "horizon": PRED_HORIZON,
        "device": str(DEVICE),
        "samples": len(X),
        "version": "2.3.6"
    }
    with open("/kaggle/working/price_action_lstm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("=" * 60)
    print("AGENT A COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
