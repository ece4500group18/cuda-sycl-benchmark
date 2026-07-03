#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n,123)
    coeff = np.array([0.25,0.5,0.75,1.0], dtype=np.float32)
    idx = np.arange(n)
    return coeff[idx & 3] * x + coeff[(idx + 1) & 3]

if __name__ == "__main__":
    V.run(reference)
