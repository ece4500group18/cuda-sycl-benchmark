#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    lut = np.array([-1.0,-0.696,-0.525,-0.394,-0.284,-0.184,-0.091,0.0,0.079,0.161,0.246,0.338,0.441,0.563,0.723,1.0], dtype=np.float32)
    q = np.floor(V.F32(16.0) * V.gen_hash01(n, 123)).astype(np.int32)
    q = np.clip(q, 0, 15)
    return lut[q]

if __name__ == "__main__":
    V.run(reference)
