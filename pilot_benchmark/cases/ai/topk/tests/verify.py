#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""topk: top-8 per row, sorted desc."""


def reference(meta):
    rows, cols = meta["input"]["sizes"]
    k = 8
    x = V.gen_hash01(rows * cols, 123).reshape(rows, cols)
    return np.sort(x, axis=1)[:, ::-1][:, :k]


if __name__ == "__main__":
    V.run(reference)
