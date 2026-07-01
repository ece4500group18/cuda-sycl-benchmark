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
    c = np.arange(cols, dtype=np.float32)
    for r in range(rows):
        slope = V.F32(0.01) * V.F32((r % 8) + 1)
        y[r] = x[r] - slope * np.abs(c - V.F32(r % cols))
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
