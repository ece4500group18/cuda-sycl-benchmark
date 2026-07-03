#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    in_n, hid_n = meta["input"]["sizes"]
    x = V.gen_hashsigned(in_n, 123)
    w = (V.F32(0.01) * V.gen_hashsigned(hid_n * (in_n + 1), 321)).reshape(hid_n, in_n + 1)
    s = w[:, 0] + w[:, 1:] @ x
    return V.F32(1.0) / (V.F32(1.0) + np.exp(-s, dtype=np.float32))

if __name__ == "__main__":
    V.run(reference)
