#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(2.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    r = (V.F32(0.25) * V.gen_hashsigned(rows * cols, 321)).reshape(rows, cols)
    g = V.F32(1.0) + V.F32(0.1) * V.gen_hashsigned(cols, 55)
    b = V.F32(0.01) * V.gen_hashsigned(cols, 66)
    v = x + r
    mean = np.mean(v, axis=1, dtype=np.float32)
    var = np.mean(v * v, axis=1, dtype=np.float32) - mean * mean
    return ((v - mean.reshape(rows, 1)) / np.sqrt(var.reshape(rows, 1) + V.F32(1.0e-5), dtype=np.float32) * g.reshape(1, cols) + b.reshape(1, cols)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
