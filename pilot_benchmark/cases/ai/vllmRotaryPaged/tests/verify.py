#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(tokens * heads * dim, 123).reshape(tokens, heads, dim)
    y = np.empty_like(x)
    for t in range(tokens):
        for h in range(heads):
            pos = (t * 37 + h * 11) % 4096
            for p in range(dim // 2):
                theta = V.F32(pos) * np.float32(10000.0) ** np.float32(-2.0 * p / dim)
                c = np.cos(theta, dtype=np.float32)
                s = np.sin(theta, dtype=np.float32)
                a = x[t, h, 2 * p]
                b = x[t, h, 2 * p + 1]
                y[t, h, 2 * p] = a * c - b * s
                y[t, h, 2 * p + 1] = a * s + b * c
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
