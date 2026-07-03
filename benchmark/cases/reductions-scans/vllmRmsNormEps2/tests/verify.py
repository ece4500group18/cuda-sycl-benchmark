#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.empty_like(x)
    for r in range(rows):
        ss = V.F32(0.0)
        for c in range(cols):
            ss += x[r, c] * x[r, c]
        inv = V.F32(1.0) / np.sqrt(ss / V.F32(cols) + V.F32(1.0e-5), dtype=np.float32)
        y[r] = x[r] * inv
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
