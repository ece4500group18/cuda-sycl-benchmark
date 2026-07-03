#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    A = (V.F32(0.01) + V.F32(0.1) * V.gen_hash01((rows+1)*cols,123)).reshape(rows+1, cols)
    B = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        B[r] = A[r+1] - A[r+1,0] * A[0]
    return B.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
