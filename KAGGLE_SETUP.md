# Kaggle Notebook Setup Guide v2.3.2

## How To Create The 4 Kaggle Notebooks

You have 4 Python files in `kaggle/agents/` and `kaggle/meta/`.
You must create **4 separate Kaggle notebooks** and paste the code into each.

---

## Dataset Path Discovery (v2.2+)

All notebooks now include an **inline robust data loader** that automatically discovers the dataset across all known Kaggle mount paths:

```
/kaggle/input/datasets/chamberbot/forex-raw-data/     ← "Add Data" via search
/kaggle/input/forex-raw-data/                          ← Legacy mount
/kaggle/input/chamberbot-forex-raw-data/               ← Owner-prefixed variant
./forex_features.parquet                                ← Local/GitHub Actions
/kaggle/working/forex_etl_output/                       ← ETL output fallback
```

If the dataset is mounted in an unexpected location, the loader **recursively scans `/kaggle/input/`** and finds it automatically. You will see debug output like:

```
[data_loader] Dataset discovery:
    FOUND: /kaggle/input/datasets/chamberbot/forex-raw-data/forex_features.parquet
    missing: /kaggle/input/forex-raw-data/forex_features.parquet
[data_loader] Loading from: /kaggle/input/datasets/chamberbot/forex-raw-data/forex_features.parquet
[data_loader] Shape: (45000, 18) | Pairs: 8 | TFs: 4
[data_loader] Validation passed: 45000 rows
```

---

## Step 1: ETL Notebook (already done)
- File: `kaggle/etl/daily_etl.py`
- Name on Kaggle: `forex-daily-etl`
- Schedule: Daily at 00:00 UTC
- Output Dataset: `forex-raw-data`

---

## Step 2: Agent A — LSTM Price Action

1. Go to https://www.kaggle.com/code → **New Notebook**
2. **Settings** (right panel):
   - **Accelerator**: GPU (T4 x2 or P100)
   - **Internet**: ON
3. **Add Data** (right panel):
   - Search: `chamberbot/forex-raw-data`
   - Click **Add**
4. Paste entire contents of `kaggle/agents/lstm_price_action.py`
5. **File → Save Version** → Save & Run
6. Check **Output** tab for `price_action_lstm.onnx`

---

## Step 3: Agent B — XGBoost Macro-Sentiment (v2.3.2)

**IMPORTANT**: This notebook uses a **stable, callback-free XGBoost API** compatible with all Kaggle XGBoost builds.

1. New Notebook
2. **Settings**: Accelerator = None (CPU), Internet = ON
3. **Add Data**: `chamberbot/forex-raw-data`
4. Paste `kaggle/agents/xgb_macro_sentiment.py`
5. Save & Run
6. Expected output: `macro_sentiment_xgb.onnx`

**What makes v2.3.2 stable:**
- No `early_stopping_rounds` parameter
- No `callbacks` parameter
- Pure `model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)`
- Fault-tolerant macro column handling (survives missing DXY/VIX/yields)

---

## Step 4: Agent C — RF Volatility Regime

1. New Notebook
2. **Settings**: Accelerator = None (CPU), Internet = ON
3. **Add Data**: `chamberbot/forex-raw-data`
4. Paste `kaggle/agents/rf_volatility_regime.py`
5. Save & Run
6. Expected output: `volatility_regime_rf.onnx`

---

## Step 5: Meta-Ensemble

1. New Notebook
2. **Settings**: Accelerator = None (CPU), Internet = ON
3. **Add Data**: `chamberbot/forex-raw-data`
4. Paste `kaggle/meta/ensemble.py`
5. Save & Run
6. Expected output: `meta_ensemble.onnx`

---

## Notebook Naming Convention

| Notebook File | Kaggle Notebook Name |
|---------------|---------------------|
| `daily_etl.py` | `forex-daily-etl` |
| `lstm_price_action.py` | `forex-lstm-price-action` |
| `xgb_macro_sentiment.py` | `forex-xgb-macro-sentiment` |
| `rf_volatility_regime.py` | `forex-rf-volatility-regime` |
| `ensemble.py` | `forex-meta-ensemble` |

---

## Troubleshooting

**"Dataset not found" even with Add Data:**
The loader will scan `/kaggle/input/` recursively. Check the debug output — it lists every path checked and the actual contents of `/kaggle/input/`. If the dataset is mounted under a different name, copy that path and add it to the `POSSIBLE_PATHS` list in the notebook.

**"Module not found" errors:**
```python
!pip install package_name -q
```

**GPU out of memory:** Reduce `BATCH_SIZE` or `SEQ_LEN` in LSTM agent.

**ONNX export fails:** Ensure `opset_version=14` and model is on CPU before export.

**XGBoost TypeError about early_stopping or callbacks:**
You are using an old version of the notebook. Update to v2.3.2 which removes these parameters entirely.
