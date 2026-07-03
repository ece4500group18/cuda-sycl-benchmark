#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H, W = meta["input"]["sizes"]
    x = V.gen_hash01(H * W, 123).reshape(H, W)
    y = np.empty((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            sr = min(max(int(np.floor(V.F32(0.75) * V.F32(r) + V.F32(4.0))), 0), H - 1)
            sc = min(max(int(np.floor(V.F32(0.75) * V.F32(c) + V.F32(2.0))), 0), W - 1)
            y[r, c] = x[sr, sc]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
