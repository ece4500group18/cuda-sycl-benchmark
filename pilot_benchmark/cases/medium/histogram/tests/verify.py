#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""histogram: 256-bin counts."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    idx = V.gen_index(n, 256, 123)
    return np.bincount(idx, minlength=256).astype(np.float32)


if __name__ == "__main__":
    V.run(reference)
