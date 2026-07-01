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
            if (((r // 16) * 7 + (c // 16) * 3) & 3) != 0:
                y[r, c] = x[r, c]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
