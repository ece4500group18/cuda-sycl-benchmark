#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    blocks, block = meta["input"]["sizes"]
    g = (V.F32(8.0) * V.gen_hashsigned(blocks * block, 123)).reshape(blocks, block)
    thr = V.F32(0.7) * np.max(np.abs(g), axis=1).astype(np.float32)
    return np.minimum(thr.reshape(blocks, 1), np.maximum(-thr.reshape(blocks, 1), g)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
