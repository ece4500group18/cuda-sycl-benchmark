#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    b = V.F32(0.5) * V.gen_hashsigned(n, 456)
    i = np.arange(n)
    xm = x[np.maximum(i - 1, 0)]
    xp = x[np.minimum(i + 1, n - 1)]
    return b - (V.F32(2.0) * x - xm - xp)

if __name__ == "__main__":
    V.run(reference)
