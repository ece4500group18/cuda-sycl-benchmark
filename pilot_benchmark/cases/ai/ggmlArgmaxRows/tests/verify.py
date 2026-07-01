#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols) + V.F32(0.0001) * np.arange(cols, dtype=np.float32).reshape(1, cols)
    return np.argmax(x, axis=1).astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
