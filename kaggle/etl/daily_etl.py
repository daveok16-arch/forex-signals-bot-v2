#!/usr/bin/env python3
"""
Forex Daily ETL - Kaggle Notebook Source (FIXED v1.2)
Run: Daily at 00:00 UTC via Kaggle Scheduler
Output: Updates Kaggle Dataset 'chamberbot/forex-raw-data'

FIXES APPLIED:
- All Datetime columns normalized to timezone-naive before merge_asof()
- Robust column flattening for yfinance multi-index outputs
- Explicit validation and logging for malformed timestamps
- Graceful handling of missing macro data
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
}

TIMEFRAMES = {
    "15m": {"interval": "15m", "period": "30d"},
    "1h": {"interval": "1h", "period": "60d"},
    "4h": {"interval": "4h", "period": "120d"},
    "1d": {"interval": "1d", "period": "180d"},
}

MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "US10Y": "^TNX",
    "DE10Y": "DE10Y.DE",
}

OUTPUT_DIR = "/kaggle/working/forex_etl_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# UTILITY: Timezone normalization
# ============================================================
def normalize_datetime(df, col_name="Datetime"):
    """
    Ensure datetime column exists, is tz-naive, and is properly typed.
    Handles yfinance inconsistencies across FX, futures, crypto, and macro.
    """
    if df is None or df.empty:
        return df

    # yfinance sometimes returns 'Date' for daily, 'Datetime' for intraday
    date_cols = [c for c in df.columns if c.lower() in ["date", "datetime"]]

    if not date_cols:
        # If index is DatetimeIndex, reset it
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_cols = [c for c in df.columns if c.lower() in ["date", "datetime"]]
        else:
            raise ValueError("No datetime column found in dataframe")

    # Standardize to 'Datetime' column name
    src_col = date_cols[0]
    if src_col != col_name:
        df = df.rename(columns={src_col: col_name})

    # Convert to datetime and strip timezone
    df[col_name] = pd.to_datetime(df[col_name], errors="coerce")

    # Remove timezone info to ensure merge_asof compatibility
    if df[col_name].dt.tz is not None:
        df[col_name] = df[col_name].dt.tz_localize(None)

    # Drop rows where datetime conversion failed
    before = len(df)
    df = df.dropna(subset=[col_name])
    after = len(df)
    if before != after:
        print(f"      [WARN] Dropped {before-after} rows with invalid timestamps")

    return df

# ============================================================
# UTILITY: Column standardization
# ============================================================
def standardize_ohlc_columns(df):
    """
    yfinance returns multi-level columns for some tickers, single-level for others.
    Flatten and standardize to Title Case: Open, High, Low, Close, Volume.
    """
    if df is None or df.empty:
        return df

    # Flatten multi-index columns
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            new_cols.append(c[0])
        else:
            new_cols.append(c)
    df.columns = new_cols

    # Standardize to Title Case
    rename_map = {}
    for c in df.columns:
        if isinstance(c, str):
            title = c.title()
            if title in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]:
                rename_map[c] = title

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["Open", "High", "Low", "Close"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns after standardization: {missing}")

    return df

# ============================================================
# DATA INGESTION
# ============================================================
def fetch_ohlc(symbol, interval, period):
    """Fetch OHLCV from Yahoo Finance with robust normalization."""
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"      [WARN] Empty dataframe for {symbol} {interval}")
            return None

        df = standardize_ohlc_columns(df)
        df = normalize_datetime(df, col_name="Datetime")
        df["symbol"] = symbol
        df["timeframe"] = interval

        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol} {interval}: {e}")
        return None

def fetch_macro():
    """Fetch macro indicators with robust normalization."""
    macro_data = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, period="180d", interval="1d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"[WARN] Macro empty for {name} ({ticker})")
                continue

            df = standardize_ohlc_columns(df)
            df = normalize_datetime(df, col_name="Datetime")

            # Validate we have data
            if len(df) < 5:
                print(f"[WARN] Macro insufficient data for {name}: {len(df)} rows")
                continue

            macro_data[name] = df
            print(f"      Macro {name}: {len(df)} rows | {df['Datetime'].min()} to {df['Datetime'].max()}")
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
    """
    Merge macro indicators using merge_asof.
    ALL dataframes must have tz-naive Datetime columns before entering.
    """
    if not macro:
        return df

    # Validate dtypes before merge
    assert df["Datetime"].dtype == "datetime64[ns]", f"FX df has wrong dtype: {df['Datetime'].dtype}"

    if "DXY" in macro:
        try:
            dxy = macro["DXY"][["Datetime", "Close"]].rename(columns={"Close": "dxy_index"})
            assert dxy["Datetime"].dtype == "datetime64[ns]", f"DXY has wrong dtype: {dxy['Datetime'].dtype}"
            df = pd.merge_asof(
                df.sort_values("Datetime"),
                dxy.sort_values("Datetime"),
                on="Datetime",
                direction="backward"
            )
            print(f"      Merged DXY: {df['dxy_index'].notna().sum()}/{len(df)} matched")
        except Exception as e:
            print(f"      [WARN] DXY merge failed: {e}")

    if "VIX" in macro:
        try:
            vix = macro["VIX"][["Datetime", "Close"]].rename(columns={"Close": "vix_proxy"})
            assert vix["Datetime"].dtype == "datetime64[ns]"
            df = pd.merge_asof(
                df.sort_values("Datetime"),
                vix.sort_values("Datetime"),
                on="Datetime",
                direction="backward"
            )
            print(f"      Merged VIX: {df['vix_proxy'].notna().sum()}/{len(df)} matched")
        except Exception as e:
            print(f"      [WARN] VIX merge failed: {e}")

    if "US10Y" in macro and "DE10Y" in macro:
        try:
            us10 = macro["US10Y"][["Datetime", "Close"]].rename(columns={"Close": "us10y"})
            de10 = macro["DE10Y"][["Datetime", "Close"]].rename(columns={"Close": "de10y"})
            assert us10["Datetime"].dtype == "datetime64[ns]"
            assert de10["Datetime"].dtype == "datetime64[ns]"

            df = pd.merge_asof(
                df.sort_values("Datetime"),
                us10.sort_values("Datetime"),
                on="Datetime",
                direction="backward"
            )
            df = pd.merge_asof(
                df.sort_values("Datetime"),
                de10.sort_values("Datetime"),
                on="Datetime",
                direction="backward"
            )
            df["yield_spread_us_de"] = df["us10y"] - df["de10y"]
            df = df.drop(columns=["us10y", "de10y"], errors="ignore")
            print(f"      Merged yields: {df['yield_spread_us_de'].notna().sum()}/{len(df)} matched")
        except Exception as e:
            print(f"      [WARN] Yield merge failed: {e}")

    return df

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("FOREX DAILY ETL PIPELINE v1.2")
    print(f"Execution: {datetime.now(timezone.utc).isoformat()} UTC")
    print("=" * 60)

    all_data = []

    # 1. Fetch macro first
    print("[1/4] Fetching macro indicators...")
    macro = fetch_macro()
    print(f"      Macro sources loaded: {list(macro.keys())}")

    # 2. Fetch all pairs across timeframes
    print("[2/4] Fetching FX pairs...")
    for alias, symbol in PAIRS.items():
        for tf_name, tf_cfg in TIMEFRAMES.items():
            print(f"      {alias} @ {tf_name}...", end=" ")
            try:
                df = fetch_ohlc(symbol, tf_cfg["interval"], tf_cfg["period"])
                if df is not None:
                    df = compute_technical_features(df)
                    df = merge_macro(df, macro, tf_name)
                    df["pair"] = alias
                    all_data.append(df)
                    print(f"OK ({len(df)} rows)")
                else:
                    print("FAIL (no data)")
            except Exception as e:
                print(f"FAIL ({e})")

    if not all_data:
        raise RuntimeError("No data fetched from any pair. Aborting.")

    # 3. Combine and save
    print("[3/4] Combining and saving...")
    combined = pd.concat(all_data, ignore_index=True)

    # Ensure no timezone leaks in final output
    if "Datetime" in combined.columns and combined["Datetime"].dt.tz is not None:
        combined["Datetime"] = combined["Datetime"].dt.tz_localize(None)

    # Save parquet (efficient) and CSV (human-readable)
    combined.to_parquet(os.path.join(OUTPUT_DIR, "forex_features.parquet"), index=False)
    combined.to_csv(os.path.join(OUTPUT_DIR, "forex_features.csv"), index=False)

    # Save metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": list(PAIRS.keys()),
        "timeframes": list(TIMEFRAMES.keys()),
        "total_rows": len(combined),
        "features": [c for c in combined.columns if c not in ["Datetime", "symbol", "timeframe", "pair"]]
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      Saved: {len(combined)} rows, {len(combined.columns)} features")
    print(f"      Datetime range: {combined['Datetime'].min()} to {combined['Datetime'].max()}")
    print(f"      Datetime dtype: {combined['Datetime'].dtype}")

    # 4. Kaggle Dataset Update
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        print("[4/4] Publishing to Kaggle Dataset...")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

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
            err = str(e).lower()
            if "already exists" in err or "resource already exists" in err:
                api.dataset_create_version(
                    folder=OUTPUT_DIR,
                    notes=f"ETL v1.2 update {datetime.now(timezone.utc).isoformat()}"
                )
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
