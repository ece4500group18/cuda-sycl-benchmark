#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    m = (np.arange(n, dtype=np.int64)*17 + 13) % n
    x = V.gen_hashsigned(n,123)
    return x[m]

if __name__ == "__main__":
    V.run(reference)
