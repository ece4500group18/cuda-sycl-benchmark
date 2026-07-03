#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    q = np.rint(V.F32(1000000.0) * x).astype(np.int32)
    return np.array([q.min() / V.F32(1000000.0), q.max() / V.F32(1000000.0)], dtype=np.float32)

if __name__ == "__main__":
    V.run(reference)
