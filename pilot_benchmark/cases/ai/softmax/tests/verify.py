#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""softmax: stable row-wise softmax."""


def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hash01(rows * cols, 123).reshape(rows, cols).astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    V.run(reference)
