#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(-1, 3)
    y = np.empty_like(x)
    y[:,0] = V.F32(0.9) * x[:,0] + V.F32(0.05) * x[:,1] + V.F32(0.02) * x[:,2]
    y[:,1] = V.F32(0.04) * x[:,0] + V.F32(1.1) * x[:,1] + V.F32(0.03) * x[:,2]
    y[:,2] = V.F32(0.02) * x[:,0] + V.F32(0.04) * x[:,1] + V.F32(0.95) * x[:,2]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
