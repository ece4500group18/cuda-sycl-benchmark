#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""gemm: C = A*B + bias."""


def reference(meta):
    N = meta["input"]["sizes"][0]
    A = V.gen_hash01(N * N, 123).reshape(N, N).astype(np.float64)
    B = V.gen_hash01(N * N, 321).reshape(N, N).astype(np.float64)
    bias = V.gen_hash01(N, 7).astype(np.float64)
    return (A @ B) + bias[None, :]


if __name__ == "__main__":
    V.run(reference)
