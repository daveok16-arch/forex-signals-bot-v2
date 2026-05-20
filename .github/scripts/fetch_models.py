#!/usr/bin/env python3
"""Download latest model artifacts from Kaggle notebook outputs."""

import os
import json
import subprocess

with open("config/kaggle_notebooks.json") as f:
    registry = json.load(f)

os.makedirs("models", exist_ok=True)

# Download agent models
for agent_key, agent in registry["notebooks"]["agents"].items():
    model_file = agent["output_model"]
    print(f"Fetching {model_file} from kernel '{agent['name']}'...")
    cmd = [
        "kaggle", "kernels", "output",
        agent["name"],
        "-p", "models/",
        "--force"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: {result.stderr}")
    else:
        print(f"  OK")

# Download meta-ensemble if available
meta = registry["notebooks"].get("meta")
if meta:
    print(f"Fetching meta-ensemble from kernel '{meta['name']}'...")
    cmd = [
        "kaggle", "kernels", "output",
        meta["name"],
        "-p", "models/",
        "--force"
    ]
    subprocess.run(cmd, capture_output=True, text=True)

print("Model download complete.")
print("Models in ./models/:")
for f in os.listdir("models"):
    if f.endswith(".onnx"):
        print(f"  - {f}")
