#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(5.0) * V.gen_hashsigned(n, 123)
    u = V.F32(0.7978845608) * (x + V.F32(0.044715) * x * x * x)
    return V.F32(0.5) * x * (V.F32(1.0) + np.tanh(u, dtype=np.float32))

if __name__ == "__main__":
    V.run(reference)
