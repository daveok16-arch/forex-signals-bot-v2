#!/usr/bin/env python3

import os
import sys
import requests
from datetime import datetime, timezone

DATASET_NAME = "chamberbot/forex-raw-data"

def check():
    url = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_NAME}"

    print(f"[DEBUG] Querying dataset: {DATASET_NAME}")
    print(f"[DEBUG] URL: {url}")

    resp = requests.get(
        url,
        auth=(os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    )

    print(f"[DEBUG] HTTP Status: {resp.status_code}")

    if resp.status_code != 200:
        print(resp.text)
        sys.exit(1)

    data = resp.json()

    print(f"[DEBUG] API Response:")
    print(data)

    last_updated = data["lastUpdated"]

    print(f"[DEBUG] lastUpdated = {last_updated}")

    last_refreshed = datetime.strptime(
        last_updated,
        "%Y-%m-%dT%H:%M:%S.%fZ"
    ).replace(tzinfo=timezone.utc)

    age_hours = (
        datetime.now(timezone.utc) - last_refreshed
    ).total_seconds() / 3600

    print(f"Dataset age: {age_hours:.1f} hours")

    if age_hours > 8:
        print("DATA STALE: Triggering alert")
        sys.exit(1)

    print("DATA FRESH: Proceeding")

if __name__ == "__main__":
    check()
