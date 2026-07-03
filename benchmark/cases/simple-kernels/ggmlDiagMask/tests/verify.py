#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = x.copy()
    for r in range(rows):
        y[r, np.arange(cols) > (r % cols)] = V.F32(-10000.0)
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
