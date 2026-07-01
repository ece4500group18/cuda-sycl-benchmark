#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    i = np.arange(n, dtype=np.int32)
    a = (i * 7 + 3) & 15; b = (i * 11 + 5) & 15
    up = (i * 13) & 63; left = (i * 17) & 63; diag = (i * 19) & 63
    match = np.where(a == b, 2, -1)
    return np.maximum(diag + match, np.maximum(up - 1, left - 1)).astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
