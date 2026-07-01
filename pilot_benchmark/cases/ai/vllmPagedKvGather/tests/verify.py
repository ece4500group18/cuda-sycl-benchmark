#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    cache = V.gen_hashsigned(tokens * heads * dim, 123).reshape(heads, tokens, dim)
    y = np.empty((tokens, heads, dim), dtype=np.float32)
    for t in range(tokens):
        page = (t * 37) % tokens
        y[t] = cache[:, page, :]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
