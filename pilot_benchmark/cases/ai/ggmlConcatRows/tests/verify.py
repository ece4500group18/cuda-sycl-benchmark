#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, ca, cb = meta["input"]["sizes"]
    a = V.gen_hashsigned(rows * ca, 123).reshape(rows, ca)
    b = (V.F32(2.0) * V.gen_hashsigned(rows * cb, 321)).reshape(rows, cb)
    return np.concatenate([a, b], axis=1).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
