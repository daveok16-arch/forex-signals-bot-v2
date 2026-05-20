#!/usr/bin/env python3
"""
Meta-Ensemble: Learns optimal combination of 3 agents v2.2
Trains on validation-set outputs from Agent A, B, C.
Output: meta_ensemble.onnx

PATCH: Robust dataset path discovery via inline data_loader.
"""

import os
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LR = 1e-2
BATCH_SIZE = 128

# ============================================================
# LOAD AGENT META + SIMULATE / LOAD REAL OUTPUTS
# ============================================================
def load_real_outputs():
    paths = [
        "/kaggle/input/forex-agent-outputs/agent_outputs.csv",
        "/kaggle/working/agent_outputs.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            X = df[["price_action_lstm", "macro_sentiment_xgb", "volatility_regime_rf"]].values.astype(np.float32)
            y = df["target"].values.astype(np.float32)
            return X, y
    return None, None

def simulate_agent_outputs(n_samples=8000, seed=42):
    np.random.seed(seed)
    y_true = np.random.binomial(1, 0.52, n_samples).astype(np.float32)
    a = np.clip(y_true * 0.7 + np.random.rand(n_samples) * 0.3, 0, 1)
    b = np.clip(y_true * 0.65 + np.random.rand(n_samples) * 0.35, 0, 1)
    c = np.clip(y_true * 0.60 + np.random.rand(n_samples) * 0.40, 0, 1)
    X = np.stack([a, b, c], axis=1).astype(np.float32)
    return X, y_true

# ============================================================
# PYTORCH
# ============================================================
class MetaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MetaEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_meta(model, train_loader, val_loader, epochs=100, lr=1e-2):
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy()
                all_pred.extend(pred.flatten())
                all_true.extend(yb.numpy().flatten())
        acc = accuracy_score((np.array(all_true) > 0.5), (np.array(all_pred) > 0.5))
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Val Acc: {acc:.2%}")
    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc

def export_onnx(model, path="/kaggle/working/meta_ensemble.onnx"):
    model.eval()
    dummy = torch.randn(1, 3).to(DEVICE)
    torch.onnx.export(model, dummy, path,
        input_names=["agent_scores"], output_names=["ensemble_probability"],
        dynamic_axes={"agent_scores": {0: "batch_size"}, "ensemble_probability": {0: "batch_size"}},
        opset_version=14)
    print(f"ONNX exported: {path}")
    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    test = sess.run(None, {"agent_scores": dummy.cpu().numpy()})[0]
    print(f"Verification: input {dummy.cpu().numpy()[0]} -> output {test[0,0]:.4f}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("META-ENSEMBLE: AGENT FUSION v2.2")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("[1/3] Loading agent outputs...")
    X, y = load_real_outputs()
    if X is None:
        print("No real outputs found. Using simulated bootstrap data.")
        X, y = simulate_agent_outputs(n_samples=8000)
    print(f"Samples: {len(X)} | Pos rate: {y.mean():.2%}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ds = MetaDataset(X_train, y_train)
    val_ds = MetaDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("[2/3] Training meta-learner...")
    model = MetaEnsemble()
    model, best_acc = train_meta(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    print(f"Best Val Accuracy: {best_acc:.2%}")

    w = model.net[0].weight.detach().cpu().numpy()
    print(f"Learned agent attention (Layer 1 weights): {w.mean(axis=0)}")

    print("[3/3] Exporting ONNX...")
    export_onnx(model)

    meta = {"agent": "meta_ensemble", "val_accuracy": float(best_acc),
            "input_agents": ["price_action_lstm", "macro_sentiment_xgb", "volatility_regime_rf"],
            "architecture": "3->8->4->1 with Sigmoid"}
    with open("/kaggle/working/meta_ensemble_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("META-ENSEMBLE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
