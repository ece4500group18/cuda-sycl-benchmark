#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    q = (np.floor(V.gen_hash01(rows * cols, 123) * V.F32(255.0)).astype(np.int32) - 127).astype(np.float32).reshape(rows, cols)
    scale = V.F32(0.001) + V.F32(0.01) * V.gen_hash01(rows, 77)
    return (q * scale.reshape(rows, 1)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
