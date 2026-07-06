#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

M, BATCH = 64, 1024


def reference(meta):
    n = M * BATCH
    L = (2.0 * V.gen_hash01(n, 101).astype(np.float64) - 1.0).reshape(M, BATCH)
    U = (2.0 * V.gen_hash01(n, 102).astype(np.float64) - 1.0).reshape(M, BATCH)
    D = (4.0 + V.gen_hash01(n, 103).astype(np.float64)).reshape(M, BATCH)
    R = (2.0 * V.gen_hash01(n, 104).astype(np.float64) - 1.0).reshape(M, BATCH)

    # Thomas forward sweep + back substitution, same recurrence order.
    U = U.copy(); R = R.copy()
    U[0] = U[0] / D[0]
    R[0] = R[0] / D[0]
    for i in range(1, M - 1):
        denom = D[i] - L[i] * U[i - 1]
        U[i] = U[i] / denom
        R[i] = (R[i] - L[i] * R[i - 1]) / denom
    R[M - 1] = (R[M - 1] - L[M - 1] * R[M - 2]) / (D[M - 1] - L[M - 1] * U[M - 2])
    for i in range(M - 2, -1, -1):
        R[i] = R[i] - U[i] * R[i + 1]
    return R.reshape(-1).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
