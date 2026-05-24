#!/usr/bin/env python3
"""Check if Kaggle forex dataset was updated within last 8 hours."""

import os
import sys
import requests
from datetime import datetime, timezone, timedelta

KAGGLE_USER = os.environ["KAGGLE_USERNAME"]
DATASET_NAME = "chamberbot/forex-raw-data"

def check():
    url = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_NAME}"
    resp = requests.get(url, auth=(os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]))
    if resp.status_code != 200:
        print(f"Failed to query dataset: {resp.status_code}")
        sys.exit(1)
    data = resp.json()
    last_refreshed = datetime.strptime(data["lastUpdated"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - last_refreshed).total_seconds() / 3600
    print(f"Dataset age: {age_hours:.1f} hours")
    if age_hours > 8:
        print("DATA STALE: Triggering alert")
        sys.exit(1)
    print("DATA FRESH: Proceeding")

if __name__ == "__main__":
    check()
