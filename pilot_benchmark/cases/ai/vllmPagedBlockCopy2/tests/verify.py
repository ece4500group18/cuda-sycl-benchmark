#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    pages, block, dim = meta["input"]["sizes"]
    src = V.gen_hashsigned(pages * block * dim, 123).reshape(pages, block, dim)
    dst = np.zeros_like(src)
    for p in range(pages):
        q = (p * 53 + 7) % pages
        dst[q] = src[p]
    return dst.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
