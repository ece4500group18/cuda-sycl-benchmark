#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""finiteDiff1d: central difference, clamped."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hash01(n, 123)
    i = np.arange(n)
    return np.float32(0.5) * (x[np.clip(i + 1, 0, n - 1)] - x[np.clip(i - 1, 0, n - 1)])


if __name__ == "__main__":
    V.run(reference)
