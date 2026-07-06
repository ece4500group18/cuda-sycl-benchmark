#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

M = N = K = 64
ALPHA, BETA = 2, 3


def reference(meta):
    A = np.floor(V.gen_hash01(M * K, 151).astype(np.float64) * 16).astype(np.int64).reshape(M, K)
    Blin = np.floor(V.gen_hash01(K * N, 152).astype(np.float64) * 16).astype(np.int64).reshape(N, K)
    B = Blin.T                       # matrix_b is col-major: B[k, n] = lin[n*K + k]
    C = np.floor(V.gen_hash01(M * N, 153).astype(np.float64) * 64).astype(np.int64).reshape(M, N)
    D = ALPHA * (A @ B) + BETA * C   # exact int32 math
    return D.reshape(-1).astype(np.float64).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
