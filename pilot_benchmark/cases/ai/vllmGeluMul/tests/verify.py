#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    import math
    n = meta["input"]["sizes"][0]
    x = V.F32(5.0) * V.gen_hashsigned(n, 123)
    g = V.F32(2.0) * V.gen_hashsigned(n, 321)
    gelu = V.F32(0.5) * x * (V.F32(1.0) + np.vectorize(math.erf, otypes=[np.float32])(x * V.F32(0.70710678118)).astype(np.float32))
    return gelu * g

if __name__ == "__main__":
    V.run(reference)
