#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""cublasAxpy: y = 2*x + y."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    return np.float32(2.0) * V.gen_hash01(n, 123) + V.gen_hash01(n, 321)


if __name__ == "__main__":
    V.run(reference)
