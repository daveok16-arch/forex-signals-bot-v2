#!/usr/bin/env python3
"""
Shared Kaggle Dataset Loader v2.2
Robust multi-path resolver for forex-raw-data across Kaggle mount variations,
local GitHub Actions runs, and manual downloads.

Usage:
    from data_loader import load_forex_data
    df = load_forex_data()
"""

import os
import sys
import pandas as pd
from typing import Optional, List

__version__ = "2.2.0"

# ============================================================
# PATH RESOLUTION
# ============================================================
DATASET_NAME = "forex-raw-data"
OWNER = "chamberbot"

# Kaggle mounts datasets in multiple possible locations depending on:
# - How the dataset was added ("Add Data" vs API vs kernel output)
# - Kaggle platform version
# - Whether running as notebook vs script
POSSIBLE_PATHS: List[str] = [
    # Kaggle "Add Data" via dataset search (most common)
    f"/kaggle/input/datasets/{OWNER}/{DATASET_NAME}/forex_features.parquet",
    f"/kaggle/input/datasets/{OWNER}/{DATASET_NAME}/forex_features.csv",
    # Kaggle legacy mount path
    f"/kaggle/input/{DATASET_NAME}/forex_features.parquet",
    f"/kaggle/input/{DATASET_NAME}/forex_features.csv",
    # Kaggle input with owner prefix (some platform versions)
    f"/kaggle/input/{OWNER}-{DATASET_NAME}/forex_features.parquet",
    f"/kaggle/input/{OWNER}-{DATASET_NAME}/forex_features.csv",
    # Local / GitHub Actions paths
    f"./forex_features.parquet",
    f"./forex_features.csv",
    f"../forex_features.parquet",
    f"../forex_features.csv",
    f"../../forex_features.parquet",
    f"../../forex_features.csv",
    # Relative to kaggle working directory
    f"/kaggle/working/forex_etl_output/forex_features.parquet",
    f"/kaggle/working/forex_etl_output/forex_features.csv",
]

# ============================================================
# CORE LOADER
# ============================================================
def load_forex_data(
    custom_paths: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load forex_features dataset with robust path discovery.

    Args:
        custom_paths: Optional additional paths to check first.
        verbose: Print debug information about path resolution.

    Returns:
        DataFrame with lowercase column names.

    Raises:
        FileNotFoundError: If dataset cannot be found in any known location.
    """
    paths_to_check = []

    if custom_paths:
        paths_to_check.extend(custom_paths)

    paths_to_check.extend(POSSIBLE_PATHS)

    # Also scan /kaggle/input/ recursively for anything matching our dataset
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for fname in files:
                if fname in ("forex_features.parquet", "forex_features.csv"):
                    full = os.path.join(root, fname)
                    if full not in paths_to_check:
                        paths_to_check.append(full)

    # Discovery phase
    found_path = None
    checked = []

    for p in paths_to_check:
        checked.append(p)
        if os.path.exists(p):
            found_path = p
            break

    if verbose:
        print(f"[data_loader v{__version__}] Dataset discovery:")
        for p in checked:
            status = "✅ FOUND" if p == found_path else "❌ missing"
            print(f"    {status}: {p}")

    if found_path is None:
        # Last resort: list everything in /kaggle/input for debugging
        debug_info = []
        if os.path.exists("/kaggle/input"):
            debug_info.append("Contents of /kaggle/input:")
            for item in os.listdir("/kaggle/input"):
                debug_info.append(f"  - {item}")
                subpath = os.path.join("/kaggle/input", item)
                if os.path.isdir(subpath):
                    try:
                        for sub in os.listdir(subpath):
                            debug_info.append(f"      - {sub}")
                    except PermissionError:
                        pass

        raise FileNotFoundError(
            f"Forex dataset not found. Checked {len(checked)} paths.\n"
            f"Last checked: {checked[-1]}\n"
            f"\n".join(debug_info)
        )

    if verbose:
        print(f"[data_loader] Loading from: {found_path}")

    # Load
    if found_path.endswith(".parquet"):
        df = pd.read_parquet(found_path)
    else:
        df = pd.read_csv(found_path)

    if verbose:
        print(f"[data_loader] Raw shape: {df.shape}")
        print(f"[data_loader] Columns: {list(df.columns)}")

    # Normalize column names to lowercase for consistency
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

    # Validate required columns exist
    required = ["close", "pair", "timeframe"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Have: {list(df.columns)}")

    if verbose:
        print(f"[data_loader] Normalized shape: {df.shape}")
        print(f"[data_loader] Pairs: {df['pair'].unique().tolist()}")
        print(f"[data_loader] Timeframes: {df['timeframe'].unique().tolist()}")

    return df


def validate_dataset(df: pd.DataFrame, min_rows: int = 1000) -> None:
    """
    Validate dataset before training begins.
    Raises RuntimeError if dataset is insufficient.
    """
    issues = []

    if len(df) < min_rows:
        issues.append(f"Only {len(df)} rows (min required: {min_rows})")

    if df["pair"].nunique() < 2:
        issues.append(f"Only {df['pair'].nunique()} unique pairs")

    if df["timeframe"].nunique() < 1:
        issues.append("No timeframe data found")

    # Check for all-NaN columns
    nan_cols = [c for c in df.columns if df[c].isna().all()]
    if nan_cols:
        issues.append(f"All-NaN columns: {nan_cols}")

    if issues:
        raise RuntimeError(
            f"Dataset validation failed ({len(issues)} issues):\n" +
            "\n".join(f"  - {i}" for i in issues)
        )

    print(f"[data_loader] Validation passed: {len(df)} rows, {df['pair'].nunique()} pairs")


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================
def load_data() -> pd.DataFrame:
    """Legacy alias for load_forex_data()."""
    return load_forex_data()


if __name__ == "__main__":
    # Self-test when run standalone
    print("Running data_loader self-test...")
    try:
        df = load_forex_data()
        validate_dataset(df)
        print("Self-test PASSED")
    except FileNotFoundError as e:
        print(f"Self-test SKIPPED (no dataset in environment):\n{e}")
