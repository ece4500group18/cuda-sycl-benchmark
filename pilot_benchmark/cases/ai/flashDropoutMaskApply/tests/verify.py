#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(2.0) * V.gen_hashsigned(n, 123)
    keep = (V.gen_hash01(n, 999) > V.F32(0.1)).astype(np.float32)
    return x * keep / V.F32(0.9)

if __name__ == "__main__":
    V.run(reference)
