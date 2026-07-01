#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.zeros_like(x)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            keep = (((idx * 1103515245 + 12345) & 7) != 0)
            if c <= (r % cols) and keep:
                y[r, c] = x[r, c] * V.F32(1.142857142857)
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
