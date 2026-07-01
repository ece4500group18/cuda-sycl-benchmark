#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H,W = meta["input"]["sizes"]
    t = V.gen_hashsigned(H*W,123).reshape(H,W)
    p = V.gen_hash01(H*W,321).reshape(H,W)
    y = np.empty_like(t)
    for r in range(H):
        for c in range(W):
            y[r,c] = t[r,c] + V.F32(0.05)*(t[max(r-1,0),c]+t[min(r+1,H-1),c]+t[r,max(c-1,0)]+t[r,min(c+1,W-1)]-V.F32(4.0)*t[r,c]) + V.F32(0.01)*p[r,c]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
