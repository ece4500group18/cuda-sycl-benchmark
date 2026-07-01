#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123); g = V.gen_hashsigned(n, 321)
    s = V.F32(100.0) * np.exp(V.F32(0.02) * x).astype(np.float32)
    avg = V.F32(0.5) * (s + V.F32(100.0) * (V.F32(1.0) + V.F32(0.01) * g))
    return np.maximum(avg - V.F32(100.0), V.F32(0.0))

if __name__ == "__main__":
    V.run(reference)
