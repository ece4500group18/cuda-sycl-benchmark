#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    p = V.gen_hashsigned(n, 11)
    g = V.gen_hashsigned(n, 22)
    m = np.rint(V.F32(60.0) * V.gen_hashsigned(n, 33)).astype(np.int8).astype(np.float32) * V.F32(0.002)
    v = np.abs(np.rint(V.F32(60.0) * V.gen_hashsigned(n, 44)).astype(np.int8).astype(np.float32)) * V.F32(0.0002) + V.F32(0.001)
    m2 = V.F32(0.9) * m + V.F32(0.1) * g
    v2 = V.F32(0.999) * v + V.F32(0.001) * g * g
    return p - V.F32(0.001) * m2 / (np.sqrt(v2, dtype=np.float32) + V.F32(1.0e-8))

if __name__ == "__main__":
    V.run(reference)
