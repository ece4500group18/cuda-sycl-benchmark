#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

M = 64


def reference(meta):
    # Column-major A, matching the harness init loop
    lin = 2.0 * V.gen_hash01(M * M, 31).astype(np.float64) - 1.0
    a = lin.reshape(M, M).T.copy()          # a[row, col], col-major fill
    a[np.arange(M), np.arange(M)] += M
    b = 2.0 * V.gen_hash01(M, 32).astype(np.float64) - 1.0
    x = np.linalg.solve(a, b)
    return x.astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
