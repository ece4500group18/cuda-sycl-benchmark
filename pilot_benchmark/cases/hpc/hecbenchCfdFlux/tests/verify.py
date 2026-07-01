#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123); g = V.gen_hashsigned(n, 321)
    rho = np.abs(x).astype(np.float32) + V.F32(1.0)
    e = V.F32(0.5) * rho + V.F32(0.25) * g * g
    return g + V.F32(0.1) * (e / rho) + np.sqrt(rho).astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
