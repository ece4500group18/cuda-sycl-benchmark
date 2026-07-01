#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(1.0) * V.gen_hashsigned(n, 123)
    g = V.F32(1.0) * V.gen_hashsigned(n, 321)
    return x - V.F32(0.001) * (V.F32(0.9) * x + V.F32(0.1) * g) / (np.sqrt(V.F32(0.99) * x * x + V.F32(0.01) * g * g).astype(np.float32) + V.F32(1.0e-6)) - V.F32(0.0001) * x

if __name__ == "__main__":
    V.run(reference)
