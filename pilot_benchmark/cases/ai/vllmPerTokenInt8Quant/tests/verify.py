#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(4.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    scale = np.max(np.abs(x), axis=1).astype(np.float32) / V.F32(127.0) + V.F32(1.0e-12)
    q = np.rint(x / scale.reshape(rows, 1)).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.float32)
    return (q * scale.reshape(rows, 1)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
