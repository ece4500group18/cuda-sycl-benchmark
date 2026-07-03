#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(N, H, W, C)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    inv = np.array([4.3668, 4.4643, 4.4444], dtype=np.float32)
    return ((x - mean.reshape(1,1,1,C)) * inv.reshape(1,1,1,C)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
