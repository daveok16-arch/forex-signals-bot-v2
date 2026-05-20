#!/usr/bin/env python3
"""
Forex Daily ETL - Kaggle Notebook Source
Run: Daily at 00:00 UTC via Kaggle Scheduler
Output: Updates Kaggle Dataset 'chamberbot/forex-raw-data'
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION (mirrors signals.yaml)
# ============================================================
PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "XAUUSD": "GC=F",          # Gold Futures
    "BTCUSD": "BTC-USD",       # Bitcoin USD
}

TIMEFRAMES = {
    "15m": {"interval": "15m", "period": "30d"},   # Entry timeframe
    "1h": {"interval": "1h", "period": "60d"},
    "4h": {"interval": "4h", "period": "120d"},
    "1d": {"interval": "1d", "period": "180d"},
}

MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",      # US Dollar Index proxy
    "VIX": "^VIX",          # Volatility index (equity proxy for sentiment)
    "US10Y": "^TNX",        # US 10Y yield
    "DE10Y": "DE10Y.DE",    # Germany 10Y yield proxy
}

OUTPUT_DIR = "/kaggle/working/forex_etl_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA INGESTION
# ============================================================
def fetch_ohlc(symbol, interval, period):
    """Fetch OHLCV from Yahoo Finance."""
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
        df["symbol"] = symbol
        df["timeframe"] = interval
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol} {interval}: {e}")
        return None

def fetch_macro():
    """Fetch macro indicators."""
    macro_data = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, period="180d", interval="1d", progress=False)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.reset_index()
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "Datetime"})
                macro_data[name] = df
        except Exception as e:
            print(f"[WARN] Macro fetch failed for {name}: {e}")
    return macro_data

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def compute_technical_features(df):
    """Add technical indicators to OHLC dataframe."""
    df = df.copy()

    # RSI 14
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # EMA Cross
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_20_50_cross"] = (df["ema_20"] > df["ema_50"]).astype(int)

    # Bollinger Bands position (0-1 scale)
    sma_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    df["bollinger_position"] = (df["Close"] - lower) / (upper - lower + 1e-9)

    # ATR 14
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df

def merge_macro(df, macro, timeframe):
    """Merge macro indicators based on closest timestamp."""
    if "DXY" in macro:
        dxy = macro["DXY"][["Datetime", "Close"]].rename(columns={"Close": "dxy_index"})
        df = pd.merge_asof(df.sort_values("Datetime"), dxy.sort_values("Datetime"), 
                           on="Datetime", direction="backward")
    if "VIX" in macro:
        vix = macro["VIX"][["Datetime", "Close"]].rename(columns={"Close": "vix_proxy"})
        df = pd.merge_asof(df.sort_values("Datetime"), vix.sort_values("Datetime"), 
                           on="Datetime", direction="backward")
    if "US10Y" in macro and "DE10Y" in macro:
        us10 = macro["US10Y"][["Datetime", "Close"]].rename(columns={"Close": "us10y"})
        de10 = macro["DE10Y"][["Datetime", "Close"]].rename(columns={"Close": "de10y"})
        df = pd.merge_asof(df.sort_values("Datetime"), us10.sort_values("Datetime"), 
                           on="Datetime", direction="backward")
        df = pd.merge_asof(df.sort_values("Datetime"), de10.sort_values("Datetime"), 
                           on="Datetime", direction="backward")
        df["yield_spread_us_de"] = df["us10y"] - df["de10y"]
        df = df.drop(columns=["us10y", "de10y"], errors="ignore")
    return df

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("FOREX DAILY ETL PIPELINE")
    print(f"Execution: {datetime.utcnow().isoformat()} UTC")
    print("=" * 60)

    all_data = []

    # 1. Fetch macro first
    print("[1/4] Fetching macro indicators...")
    macro = fetch_macro()

    # 2. Fetch all pairs across timeframes
    print("[2/4] Fetching FX pairs...")
    for alias, symbol in PAIRS.items():
        for tf_name, tf_cfg in TIMEFRAMES.items():
            print(f"      {alias} @ {tf_name}...", end=" ")
            df = fetch_ohlc(symbol, tf_cfg["interval"], tf_cfg["period"])
            if df is not None:
                df = compute_technical_features(df)
                df = merge_macro(df, macro, tf_name)
                df["pair"] = alias
                all_data.append(df)
                print(f"OK ({len(df)} rows)")
            else:
                print("FAIL")

    if not all_data:
        raise RuntimeError("No data fetched. Aborting.")

    # 3. Combine and save
    print("[3/4] Combining and saving...")
    combined = pd.concat(all_data, ignore_index=True)

    # Save parquet (efficient) and CSV (human-readable)
    combined.to_parquet(os.path.join(OUTPUT_DIR, "forex_features.parquet"), index=False)
    combined.to_csv(os.path.join(OUTPUT_DIR, "forex_features.csv"), index=False)

    # Save metadata
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "pairs": list(PAIRS.keys()),
        "timeframes": list(TIMEFRAMES.keys()),
        "total_rows": len(combined),
        "features": [c for c in combined.columns if c not in ["Datetime", "symbol", "timeframe", "pair"]]
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      Saved: {len(combined)} rows, {len(combined.columns)} features")

    # 4. Kaggle Dataset Update (if running on Kaggle)
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        print("[4/4] Publishing to Kaggle Dataset...")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # Create dataset metadata
        dataset_meta = {
            "title": "Forex Raw Features",
            "id": "chamberbot/forex-raw-data",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(os.path.join(OUTPUT_DIR, "dataset-metadata.json"), "w") as f:
            json.dump(dataset_meta, f)

        try:
            api.dataset_create_new(folder=OUTPUT_DIR, public=False, convert_to_csv=False)
            print("      Dataset created.")
        except Exception as e:
            if "already exists" in str(e).lower() or "resource already exists" in str(e).lower():
                api.dataset_create_version(folder=OUTPUT_DIR, notes=f"ETL update {datetime.utcnow().isoformat()}")
                print("      Dataset versioned.")
            else:
                print(f"      Dataset publish warning: {e}")
    else:
        print("[4/4] Local run - skipping Kaggle dataset publish.")

    print("=" * 60)
    print("ETL COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
