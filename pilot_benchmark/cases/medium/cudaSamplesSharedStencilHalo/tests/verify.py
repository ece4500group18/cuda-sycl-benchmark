#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    left = np.concatenate([x[:1], x[:-1]])
    right = np.concatenate([x[1:], x[-1:]])
    return V.F32(0.25) * left + V.F32(0.5) * x + V.F32(0.25) * right

if __name__ == "__main__":
    V.run(reference)
