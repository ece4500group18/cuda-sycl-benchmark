#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""stencil1d: 3-point clamped stencil."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hash01(n, 123)
    i = np.arange(n)
    im = np.clip(i - 1, 0, n - 1); ip = np.clip(i + 1, 0, n - 1)
    return np.float32(0.25) * x[im] + np.float32(0.5) * x[i] + np.float32(0.25) * x[ip]


if __name__ == "__main__":
    V.run(reference)
