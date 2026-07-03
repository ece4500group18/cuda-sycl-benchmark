#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    i = np.arange(n, dtype=np.float32)
    up = i % 17; left = i % 13; diag = i % 11
    match = diag + np.where((np.arange(n)%7)==0, 2.0, -1.0).astype(np.float32)
    return np.maximum(match, np.maximum(up - V.F32(1.0), left - V.F32(1.0))).astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
