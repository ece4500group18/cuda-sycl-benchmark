#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        m = np.max(x[r]).astype(np.float32)
        s = V.F32(0.0)
        for c in range(cols):
            s += np.exp(x[r, c] - m, dtype=np.float32)
        y[r] = s
    return y

if __name__ == "__main__":
    V.run(reference)
