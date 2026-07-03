#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(tokens * heads * dim, 123).reshape(tokens, heads, dim)
    return np.transpose(x, (1, 0, 2)).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
