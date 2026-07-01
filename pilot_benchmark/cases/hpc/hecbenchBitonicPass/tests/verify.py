#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    y = np.empty_like(x)
    for i in range(n):
        ixj = i ^ 1
        up = (i & 2) == 0
        a = x[i]
        b = x[ixj]
        y[i] = b if ((up and a > b) or ((not up) and a < b)) else a
    return y

if __name__ == "__main__":
    V.run(reference)
