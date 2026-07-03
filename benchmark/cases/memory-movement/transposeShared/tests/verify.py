#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""transposeShared: out = in^T."""


def reference(meta):
    rows, cols = meta["input"]["sizes"]
    return V.gen_hash01(rows * cols, 123).reshape(rows, cols).T


if __name__ == "__main__":
    V.run(reference)
