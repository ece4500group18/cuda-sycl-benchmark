#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 1)
    v = V.F32(0.1) * V.gen_hashsigned(n, 2)
    a = V.F32(0.01) * V.gen_hashsigned(n, 3)
    vn = v + a * V.F32(0.01)
    xn = x + vn * V.F32(0.01)
    out = np.empty(2 * n, dtype=np.float32)
    out[0::2] = xn
    out[1::2] = vn
    return out

if __name__ == "__main__":
    V.run(reference)
