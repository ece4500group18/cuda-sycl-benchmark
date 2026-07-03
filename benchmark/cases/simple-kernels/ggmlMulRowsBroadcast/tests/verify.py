#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    b = V.F32(0.5) + V.F32(0.25) * V.gen_hashsigned(cols, 321)
    return (x * b.reshape(1, cols)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
