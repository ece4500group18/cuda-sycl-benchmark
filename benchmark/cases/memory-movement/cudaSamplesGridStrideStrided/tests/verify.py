#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(2 * n, 123)
    idx = (np.arange(n, dtype=np.int64) * 2 + 17) % (2 * n)
    return x[idx]

if __name__ == "__main__":
    V.run(reference)
