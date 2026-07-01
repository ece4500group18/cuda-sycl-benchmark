#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, nnz_per = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows, 123)
    y = np.zeros(rows, dtype=np.float32)
    vals = V.F32(0.1) + V.F32(0.01) * np.arange(nnz_per, dtype=np.float32)
    for r in range(rows):
        for j in range(nnz_per):
            c = (r + j * 13 + rows - 39) % rows
            y[r] += vals[j] * x[c]
    return y

if __name__ == "__main__":
    V.run(reference)
