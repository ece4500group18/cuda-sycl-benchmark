#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""spmv: 1D Laplacian (diag 2, offdiag -1) times x."""


def reference(meta):
    N = meta["input"]["sizes"][0]
    x = V.gen_hash01(N, 123)
    y = np.float32(2.0) * x.copy()
    y[1:] -= x[:-1]
    y[:-1] -= x[1:]
    return y


if __name__ == "__main__":
    V.run(reference)
