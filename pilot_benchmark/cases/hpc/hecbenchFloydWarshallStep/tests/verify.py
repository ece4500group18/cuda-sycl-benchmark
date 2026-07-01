#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]; k = 37
    x = ((np.arange(n*n, dtype=np.int64)*17 + 23) % 251).astype(np.float32).reshape(n,n)
    np.fill_diagonal(x, 0.0)
    return np.minimum(x, x[:,[k]] + x[[k],:]).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
