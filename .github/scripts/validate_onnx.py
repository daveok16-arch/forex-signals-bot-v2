#!/usr/bin/env python3
"""Validate all ONNX models in models/ directory."""

import os
import glob
import onnxruntime as ort
import numpy as np

models = glob.glob("models/*.onnx")
if not models:
    print("No models found!")
    exit(1)

for path in models:
    print(f"Validating {path}...")
    try:
        sess = ort.InferenceSession(path)
        # Try a dummy inference
        for inp in sess.get_inputs():
            shape = [1 if isinstance(dim, str) or dim is None else dim for dim in inp.shape]
            dummy = np.random.randn(*shape).astype(np.float32)
            sess.run(None, {inp.name: dummy})
        print(f"  ✅ {os.path.basename(path)} OK")
    except Exception as e:
        print(f"  ❌ {os.path.basename(path)} FAILED: {e}")
        exit(1)

print("All models validated.")
