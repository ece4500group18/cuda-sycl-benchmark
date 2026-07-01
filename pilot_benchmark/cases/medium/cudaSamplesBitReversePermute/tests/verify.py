#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    y = np.empty(n, dtype=np.float32)
    for i in range(n):
        v = i
        v = ((v & 0x5555) << 1) | ((v >> 1) & 0x5555)
        v = ((v & 0x3333) << 2) | ((v >> 2) & 0x3333)
        v = ((v & 0x0f0f) << 4) | ((v >> 4) & 0x0f0f)
        v = ((v << 8) | (v >> 8)) & 0xffff
        y[v] = x[i]
    return y

if __name__ == "__main__":
    V.run(reference)
