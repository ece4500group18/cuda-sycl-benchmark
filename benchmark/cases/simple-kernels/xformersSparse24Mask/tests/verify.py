#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    groups, width = meta["input"]["sizes"]
    x = V.gen_hashsigned(groups * width, 123).reshape(groups, width)
    order = np.argsort(-np.abs(x), axis=1)
    keep = np.zeros_like(x)
    rows = np.arange(groups)
    keep[rows, order[:, 0]] = V.F32(1.0)
    keep[rows, order[:, 1]] = V.F32(1.0)
    return (x * keep).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
