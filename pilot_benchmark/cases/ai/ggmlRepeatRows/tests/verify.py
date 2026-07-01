#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, src_rows, cols = meta["input"]["sizes"]
    src = V.gen_hashsigned(src_rows * cols, 123).reshape(src_rows, cols)
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        out[r] = src[r % src_rows]
    return out.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
