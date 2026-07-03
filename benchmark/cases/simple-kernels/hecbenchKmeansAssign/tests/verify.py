#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    pts, dim, k = meta["input"]["sizes"]
    x = V.gen_hashsigned(pts*dim,123).reshape(pts,dim)
    c = V.gen_hashsigned(k*dim,321).reshape(k,dim)
    d = ((x[:,None,:]-c[None,:,:])**2).sum(axis=2, dtype=np.float32)
    return np.argmin(d, axis=1).astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
