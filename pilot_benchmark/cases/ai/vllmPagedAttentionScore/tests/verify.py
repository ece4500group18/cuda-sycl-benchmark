#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    queries, keys, dim = meta["input"]["sizes"]
    q = V.gen_hashsigned(queries * dim, 123).reshape(queries, dim)
    k = V.gen_hashsigned(keys * dim, 321).reshape(keys, dim)
    return (q @ k.T * (V.F32(1.0) / np.sqrt(V.F32(dim), dtype=np.float32))).astype(np.float32).reshape(-1)

if __name__ == "__main__":
    V.run(reference)
