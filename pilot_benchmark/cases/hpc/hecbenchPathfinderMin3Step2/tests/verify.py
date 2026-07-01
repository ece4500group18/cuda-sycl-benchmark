#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H, W = meta["input"]["sizes"]
    prev = V.gen_hash01(W, 77)
    cost = V.gen_hash01(H * W, 123).reshape(H, W)
    y = np.empty((H, W), dtype=np.float32)
    for c in range(W):
        best = min(prev[max(c - 1, 0)], prev[c], prev[min(c + 1, W - 1)])
        y[:, c] = cost[:, c] + best
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
