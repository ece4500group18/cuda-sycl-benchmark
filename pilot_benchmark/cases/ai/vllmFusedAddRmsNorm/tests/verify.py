#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(2.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    r = (V.F32(0.5) * V.gen_hashsigned(rows * cols, 777)).reshape(rows, cols)
    w = V.F32(1.0) + V.F32(0.1) * V.gen_hashsigned(cols, 44)
    v = x + r
    inv = V.F32(1.0) / np.sqrt(np.mean(v * v, axis=1, dtype=np.float32) + V.F32(1.0e-6), dtype=np.float32)
    return (v * inv.reshape(rows, 1) * w.reshape(1, cols)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
