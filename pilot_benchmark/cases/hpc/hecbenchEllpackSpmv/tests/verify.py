#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, width = meta["input"]["sizes"]
    val = (V.F32(0.1) * V.gen_hashsigned(rows * width, 123)).reshape(rows, width)
    x = V.gen_hashsigned(rows, 321)
    y = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        acc = V.F32(0.0)
        for k in range(width):
            c = (r * 17 + k * 13) % rows
            acc += val[r, k] * x[c]
        y[r] = acc
    return y

if __name__ == "__main__":
    V.run(reference)
