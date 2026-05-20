# Kaggle Notebook Setup Guide

## How To Create The 4 Kaggle Notebooks

You have 4 Python files in `kaggle/agents/` and `kaggle/meta/`. 
You must create **4 separate Kaggle notebooks** and paste the code into each.

---

## Step 1: Create the ETL Notebook (already done in Step 1)
- File: `kaggle/etl/daily_etl.py`
- Name on Kaggle: `forex-daily-etl`
- Schedule: Daily at 00:00 UTC
- Output Dataset: `forex-raw-data`

---

## Step 2: Create Agent A — LSTM Price Action

1. Go to https://www.kaggle.com/code → **New Notebook**
2. **Settings** (right panel):
   - **Accelerator**: GPU (T4 x2 or P100)
   - **Internet**: ON
3. **Add Data** (right panel):
   - Search for your dataset: `chamberbot/forex-raw-data`
   - Click **Add**
4. In the notebook, create a **Code cell** and paste the **entire contents** of:
   `kaggle/agents/lstm_price_action.py`
5. **File → Save Version** → Save & Run
6. Wait for training to complete (5-15 minutes)
7. Check **Output** tab — you should see:
   - `price_action_lstm.onnx`
   - `price_action_lstm_meta.json`
8. **Important**: Go to notebook settings and **turn ON "Keep data source in sync"**

---

## Step 3: Create Agent B — XGBoost Macro-Sentiment

1. New Notebook
2. **Settings**:
   - **Accelerator**: None (CPU is fine)
   - **Internet**: ON
3. **Add Data**: `chamberbot/forex-raw-data`
4. Paste entire contents of: `kaggle/agents/xgb_macro_sentiment.py`
5. Save & Run
6. Expected output:
   - `macro_sentiment_xgb.onnx`
   - `macro_sentiment_xgb_meta.json`

---

## Step 4: Create Agent C — RF Volatility Regime

1. New Notebook
2. **Settings**:
   - **Accelerator**: None (CPU)
   - **Internet**: ON
3. **Add Data**: `chamberbot/forex-raw-data`
4. Paste entire contents of: `kaggle/agents/rf_volatility_regime.py`
5. Save & Run
6. Expected output:
   - `volatility_regime_rf.onnx`
   - `volatility_regime_rf_meta.json`

---

## Step 5: Create Meta-Ensemble

1. New Notebook
2. **Settings**:
   - **Accelerator**: None (CPU)
   - **Internet**: ON
3. **Add Data**: `chamberbot/forex-raw-data`
4. Paste entire contents of: `kaggle/meta/ensemble.py`
5. Save & Run
6. Expected output:
   - `meta_ensemble.onnx`
   - `meta_ensemble_meta.json`

**Note**: The meta-ensemble uses simulated data on first run. For production accuracy,
save validation outputs from Agents A/B/C as `agent_outputs.csv` and re-run.

---

## Notebook Naming Convention

Name your notebooks **exactly** as follows so GitHub Actions can fetch them:

| Notebook File | Kaggle Notebook Name |
|---------------|---------------------|
| `daily_etl.py` | `forex-daily-etl` |
| `lstm_price_action.py` | `forex-lstm-price-action` |
| `xgb_macro_sentiment.py` | `forex-xgb-macro-sentiment` |
| `rf_volatility_regime.py` | `forex-rf-volatility-regime` |
| `ensemble.py` | `forex-meta-ensemble` |

---

## After All Notebooks Are Created

1. Update `config/kaggle_notebooks.json` in your GitHub repo with the real notebook IDs
2. Run the GitHub Action manually to test model fetching:
   - Go to GitHub repo → Actions → Nightly ETL → Run workflow
3. Check that models appear in GitHub Release artifacts

---

## Troubleshooting

**"Module not found" errors**: Kaggle notebooks have most packages preinstalled. If missing:
```python
!pip install package_name -q
```

**"Dataset not found"**: Make sure you clicked **Add Data** and the dataset path is:
```python
/kaggle/input/forex-raw-data/forex_features.parquet
```

**GPU out of memory**: Reduce `BATCH_SIZE` or `SEQ_LEN` in LSTM agent.

**ONNX export fails**: Ensure you are using `opset_version=14` and the model is on CPU before export.
