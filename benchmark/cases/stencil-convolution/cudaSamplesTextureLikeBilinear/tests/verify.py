#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H,W = meta["input"]["sizes"]
    x = V.gen_hash01(H*W,123).reshape(H,W)
    y = np.empty((H,W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            u = V.F32(c) + V.F32(0.35); v = V.F32(r) + V.F32(0.65)
            x0 = min(int(np.floor(u)), W-1); x1 = min(x0+1, W-1); y0 = min(int(np.floor(v)), H-1); y1 = min(y0+1, H-1)
            fx = u - V.F32(x0); fy = v - V.F32(y0)
            a = x[y0,x0]*(V.F32(1.0)-fx)+x[y0,x1]*fx
            b = x[y1,x0]*(V.F32(1.0)-fx)+x[y1,x1]*fx
            y[r,c] = a*(V.F32(1.0)-fy)+b*fy
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
