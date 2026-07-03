#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, k = meta["input"]["sizes"]
    y = np.empty(n * n, dtype=np.float32)
    for idx in range(n * n):
        i = idx // n; j = idx % n
        cur = (i * 13 + j * 7) & 1023
        via = ((i * 13 + k * 7) & 1023) + ((k * 13 + j * 7) & 1023)
        y[idx] = min(cur, via)
    return y

if __name__ == "__main__":
    V.run(reference)
