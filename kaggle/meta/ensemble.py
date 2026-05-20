#!/usr/bin/env python3
"""
Meta-Ensemble: Learns optimal combination of 3 agents.
Trains on validation-set outputs from Agent A, B, C.
Output: meta_ensemble.onnx
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
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LR = 1e-2
BATCH_SIZE = 128

# ============================================================
# LOAD AGENT META + SIMULATE VALIDATION OUTPUTS
# ============================================================
def load_agent_meta(agent_name):
    path = f"/kaggle/input/forex-raw-data/{agent_name}_meta.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def simulate_agent_outputs(n_samples=5000, seed=42):
    """
    Since we cannot run agents A/B/C inside this notebook sequentially
    in production, this notebook expects you to have saved validation
    predictions from each agent as CSVs in the dataset.

    For first-run bootstrap, we simulate correlated predictions.
    In production, replace this with actual agent validation outputs.
    """
    np.random.seed(seed)

    # Simulate 3 agents with different accuracies and correlations
    y_true = np.random.binomial(1, 0.52, n_samples).astype(np.float32)

    # Agent A (LSTM): 55% acc, correlated with true
    a = y_true * 0.7 + np.random.rand(n_samples) * 0.3
    a = np.clip(a, 0, 1)

    # Agent B (XGB): 53% acc, macro-driven
    b = y_true * 0.65 + np.random.rand(n_samples) * 0.35
    b = np.clip(b, 0, 1)

    # Agent C (RF): 51% acc, volatility
    c = y_true * 0.60 + np.random.rand(n_samples) * 0.40
    c = np.clip(c, 0, 1)

    X = np.stack([a, b, c], axis=1).astype(np.float32)
    return X, y_true

def load_real_outputs():
    """Attempt to load real agent outputs from /kaggle/working or input."""
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

# ============================================================
# PYTORCH DATASET
# ============================================================
class MetaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# MODEL: Tiny Meta-Learner
# ============================================================
class MetaEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# TRAINING
# ============================================================
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

        # Validation
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

# ============================================================
# ONNX EXPORT
# ============================================================
def export_onnx(model, path="/kaggle/working/meta_ensemble.onnx"):
    model.eval()
    dummy = torch.randn(1, 3).to(DEVICE)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["agent_scores"],
        output_names=["ensemble_probability"],
        dynamic_axes={
            "agent_scores": {0: "batch_size"},
            "ensemble_probability": {0: "batch_size"}
        },
        opset_version=14
    )
    print(f"✅ ONNX exported: {path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(path)
    test = sess.run(None, {"agent_scores": dummy.cpu().numpy()})[0]
    print(f"   Verification: input {dummy.cpu().numpy()[0]} -> output {test[0,0]:.4f}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("META-ENSEMBLE: AGENT FUSION")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("[1/3] Loading agent outputs...")
    X, y = load_real_outputs()
    if X is None:
        print("      No real outputs found. Using simulated bootstrap data.")
        print("      NOTE: For production, save agent validation outputs to dataset.")
        X, y = simulate_agent_outputs(n_samples=8000)

    print(f"      Samples: {len(X)} | Pos rate: {y.mean():.2%}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = MetaDataset(X_train, y_train)
    val_ds = MetaDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("[2/3] Training meta-learner...")
    model = MetaEnsemble()
    model, best_acc = train_meta(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)
    print(f"      Best Val Accuracy: {best_acc:.2%}")

    # Show learned weights (first layer)
    w = model.net[0].weight.detach().cpu().numpy()
    print(f"      Learned agent attention (Layer 1 weights): {w.mean(axis=0)}")

    print("[3/3] Exporting ONNX...")
    export_onnx(model)

    meta = {
        "agent": "meta_ensemble",
        "val_accuracy": float(best_acc),
        "input_agents": ["price_action_lstm", "macro_sentiment_xgb", "volatility_regime_rf"],
        "architecture": "3->8->4->1 with Sigmoid"
    }
    with open("/kaggle/working/meta_ensemble_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 60)
    print("META-ENSEMBLE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
