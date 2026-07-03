#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H,W = meta["input"]["sizes"]
    x = (V.F32(1.0) + V.gen_hash01(H*W,123)).reshape(H,W)
    y = np.empty_like(x)
    for r in range(H):
        for c in range(W):
            center = x[r,c]
            y[r,c] = center + V.F32(0.125)*((x[max(r-1,0),c]+x[min(r+1,H-1),c]+x[r,max(c-1,0)]+x[r,min(c+1,W-1)]-V.F32(4.0)*center)/(V.F32(0.01)+abs(center)))
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
