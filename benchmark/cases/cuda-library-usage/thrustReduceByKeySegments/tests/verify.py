#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, seg = meta["input"]["sizes"]
    x = (V.F32(0.01) * V.gen_hashsigned(n, 123)).reshape(-1, seg)
    y = np.zeros(n // seg, dtype=np.float32)
    for c in range(seg):
        y += x[:, c]
    return y

if __name__ == "__main__":
    V.run(reference)
