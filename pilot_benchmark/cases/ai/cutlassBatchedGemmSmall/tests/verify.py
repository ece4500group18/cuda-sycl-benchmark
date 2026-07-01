#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    batch, M, N, K = meta["input"]["sizes"]
    a = (V.F32(0.2) * V.gen_hashsigned(batch * M * K, 123)).reshape(batch, M, K)
    b = (V.F32(0.2) * V.gen_hashsigned(batch * K * N, 321)).reshape(batch, K, N)
    return np.matmul(a, b).astype(np.float32).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
