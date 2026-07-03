#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""batchedGemm: per-batch C_b = A_b*B_b."""


def reference(meta):
    batch, N = meta["input"]["sizes"]
    outs = []
    for b in range(batch):
        A = V.gen_hash01(N * N, 100 + b).reshape(N, N).astype(np.float64)
        B = V.gen_hash01(N * N, 200 + b).reshape(N, N).astype(np.float64)
        outs.append((A @ B).reshape(-1))
    return np.concatenate(outs)


if __name__ == "__main__":
    V.run(reference)
