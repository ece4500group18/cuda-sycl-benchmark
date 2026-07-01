#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    src = V.gen_hashsigned(tokens * heads * dim, 123).reshape(tokens, heads, dim)
    cache = np.zeros((heads, tokens, dim), dtype=np.float32)
    for t in range(tokens):
        page = (t * 37) % tokens
        cache[:, page, :] = src[t]
    return cache.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
