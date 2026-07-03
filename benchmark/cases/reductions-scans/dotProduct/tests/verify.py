#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""dotProduct: sum(a*b)."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    a = V.gen_hash01(n, 123).astype(np.float64)
    b = V.gen_hash01(n, 321).astype(np.float64)
    return np.array([(a * b).sum()], dtype=np.float32)


if __name__ == "__main__":
    V.run(reference)
