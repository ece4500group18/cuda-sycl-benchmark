#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H, W = meta["input"]["sizes"]
    x = V.gen_hashsigned(H * W, 123).reshape(H, W)
    y = x.copy()
    y[1:-1,1:-1] = V.F32(0.6) * x[1:-1,1:-1] + V.F32(0.1) * (x[1:-1,:-2] + x[1:-1,2:] + x[:-2,1:-1] + x[2:,1:-1])
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
