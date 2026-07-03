#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, B = meta["input"]["sizes"]
    x = (V.F32(0.01) * V.gen_hashsigned(n, 123)).reshape(-1, B)
    return np.sum(x, axis=1, dtype=np.float32)

if __name__ == "__main__":
    V.run(reference)
