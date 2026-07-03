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
            y[r, c] = x[min(max(r - 5, 0), H - 1), min(max(c + 7, 0), W - 1)]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
