#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    blocks, block = meta["input"]["sizes"]
    n = blocks * block
    q = np.rint(V.F32(127.0) * V.gen_hashsigned(n, 123)).astype(np.int8).astype(np.float32)
    scale = V.F32(0.01) + V.F32(0.05) * V.gen_hash01(blocks, 321)
    return q * np.repeat(scale, block)

if __name__ == "__main__":
    V.run(reference)
