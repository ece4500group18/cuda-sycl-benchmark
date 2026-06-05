#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""conv1dShared: radius-3 fixed-weight convolution, edge-clamped."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hash01(n, 123)
    w = (np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float32) / np.float32(16.0))
    i = np.arange(n)[:, None]
    k = np.arange(7)[None, :]
    gi = np.clip(i - 3 + k, 0, n - 1)
    return (x[gi] * w[None, :]).sum(axis=1)


if __name__ == "__main__":
    V.run(reference)
