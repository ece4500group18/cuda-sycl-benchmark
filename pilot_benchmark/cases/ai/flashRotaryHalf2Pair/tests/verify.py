#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * dim, 123).reshape(rows, dim)
    y = np.empty_like(x)
    for r in range(rows):
        for p in range(dim // 2):
            th = V.F32(r % 2048) * np.float32(10000.0) ** np.float32(-2.0 * p / dim)
            c = np.cos(th, dtype=np.float32); s = np.sin(th, dtype=np.float32)
            a = x[r, 2*p]; b = x[r, 2*p+1]
            y[r, 2*p] = a*c - b*s
            y[r, 2*p+1] = a*s + b*c
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
