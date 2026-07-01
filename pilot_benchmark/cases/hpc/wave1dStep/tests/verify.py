#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    prev = V.gen_hashsigned(n, 13)
    cur = V.gen_hashsigned(n, 14)
    i = np.arange(n)
    left = cur[np.maximum(i - 1, 0)]
    right = cur[np.minimum(i + 1, n - 1)]
    return V.F32(2.0) * cur - prev + V.F32(0.1) * (left - V.F32(2.0) * cur + right)

if __name__ == "__main__":
    V.run(reference)
