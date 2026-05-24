#!/usr/bin/env python3

import os
import json
import sys
from datetime import datetime, timezone

META_PATH = "data/metadata.json"

def is_market_closed():
    now = datetime.now(timezone.utc)

    wd = now.weekday()
    hr = now.hour

    if wd == 4 and hr >= 21:
        return True

    if wd == 5:
        return True

    if wd == 6 and hr < 21:
        return True

    return False

def main():
    if not os.path.exists(META_PATH):
        print(f"DATA STALE: {META_PATH} missing.")
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    ts_str = meta.get("timestamp_utc", "")

    if not ts_str:
        print("DATA STALE: No timestamp_utc.")
        sys.exit(1)

    last_run = datetime.fromisoformat(
        ts_str.replace("Z", "+00:00")
    )

    age_hours = (
        datetime.now(timezone.utc) - last_run
    ).total_seconds() / 3600

    threshold = 72 if is_market_closed() else 24

    print(f"Dataset age: {age_hours:.1f} hours")
    print(f"Threshold: {threshold} hours")
    print(f"Last ETL: {ts_str}")

    if age_hours > threshold:
        print("DATA STALE: Triggering alert")
        sys.exit(1)

    print("DATA FRESH")
    sys.exit(0)

if __name__ == "__main__":
    main()
