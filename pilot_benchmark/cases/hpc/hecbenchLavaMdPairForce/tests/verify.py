#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, neigh = meta["input"]["sizes"]
    x = V.gen_hashsigned(n, 123)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        acc = V.F32(0.0); xi = x[i]
        for k in range(1, neigh + 1):
            j = (i + k * 17) % n
            d = x[j] - xi
            r = V.F32(1.0) / np.sqrt(d * d + V.F32(0.01), dtype=np.float32)
            acc += d * r * r * r
        y[i] = acc
    return y

if __name__ == "__main__":
    V.run(reference)
