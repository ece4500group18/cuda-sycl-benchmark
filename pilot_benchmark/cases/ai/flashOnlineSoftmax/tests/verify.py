#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(3.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    m = np.max(x, axis=1)
    e = np.exp(x - m.reshape(rows, 1), dtype=np.float32)
    return (e / np.sum(e, axis=1, dtype=np.float32).reshape(rows, 1)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
