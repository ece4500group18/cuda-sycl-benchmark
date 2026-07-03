#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    pts, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(pts*dim,123).reshape(pts,dim)
    c = V.gen_hashsigned(dim,321)
    return np.sum((x-c.reshape(1,dim))**2, axis=1, dtype=np.float32)

if __name__ == "__main__":
    V.run(reference)
