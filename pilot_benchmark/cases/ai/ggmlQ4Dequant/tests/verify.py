#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    blocks, qk = meta["input"]["sizes"]
    n = blocks * qk
    q = np.floor(V.F32(16.0) * V.gen_hash01(n, 123)).astype(np.int32)
    q = np.clip(q, 0, 15).astype(np.float32)
    scale = V.F32(0.04) + V.F32(0.02) * V.gen_hash01(blocks, 88)
    return (q - V.F32(8.0)) * np.repeat(scale, qk)

if __name__ == "__main__":
    V.run(reference)
