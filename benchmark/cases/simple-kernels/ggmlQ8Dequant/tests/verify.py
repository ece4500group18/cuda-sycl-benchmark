#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    blocks, qk = meta["input"]["sizes"]
    n = blocks * qk
    q = np.rint(V.F32(127.0) * V.gen_hashsigned(n, 123)).astype(np.int8).astype(np.float32)
    scale = V.F32(0.02) + V.F32(0.03) * V.gen_hash01(blocks, 77)
    return q * np.repeat(scale, qk)

if __name__ == "__main__":
    V.run(reference)
