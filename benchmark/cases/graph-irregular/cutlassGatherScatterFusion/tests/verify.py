#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        src = (i * 17 + 13) % n
        dst = (i * 29 + 7) % n
        y[dst] = V.F32(2.0) * x[src]
    return y

if __name__ == "__main__":
    V.run(reference)
