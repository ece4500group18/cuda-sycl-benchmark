#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""rmsnorm: per-row RMSNorm (eps=1e-6)."""


def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hash01(rows * cols, 123).reshape(rows, cols).astype(np.float64)
    ms = (x ** 2).mean(axis=1, keepdims=True)
    return x / np.sqrt(ms + 1e-6)


if __name__ == "__main__":
    V.run(reference)
