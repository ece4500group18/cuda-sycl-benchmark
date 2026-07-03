#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(3.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    y = np.zeros_like(x)
    for r in range(rows):
        limit = r % cols
        vals = x[r, :limit + 1]
        m = np.max(vals)
        e = np.exp(vals - m, dtype=np.float32)
        y[r, :limit + 1] = e / np.sum(e, dtype=np.float32)
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
