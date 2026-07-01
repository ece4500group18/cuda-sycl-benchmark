#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    q = np.floor(V.gen_hashsigned(n, 123) * V.F32(32.0) + V.F32(128.0)).astype(np.int32)
    return np.clip(q, 0, 255).astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
