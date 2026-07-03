#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H0, W0, H1, W1 = meta["input"]["sizes"]
    x = V.gen_hash01(H0 * W0, 123).reshape(H0, W0)
    y = np.empty((H1, W1), dtype=np.float32)
    for r in range(H1):
        for c in range(W1):
            sr = min(int(np.floor((V.F32(r) + V.F32(0.5)) * V.F32(H0) / V.F32(H1))), H0 - 1)
            sc = min(int(np.floor((V.F32(c) + V.F32(0.5)) * V.F32(W0) / V.F32(W1))), W0 - 1)
            y[r, c] = x[sr, sc]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
