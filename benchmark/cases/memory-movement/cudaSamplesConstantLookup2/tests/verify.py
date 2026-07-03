#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    lut = np.array([0.125,0.25,0.5,0.75,1.0,1.25,1.5,2.0], dtype=np.float32)
    return V.gen_hashsigned(n, 123) * lut[np.arange(n) & 7]

if __name__ == "__main__":
    V.run(reference)
