#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    blocks, block = meta["input"]["sizes"]
    x = V.gen_hashsigned(blocks * block, 123).reshape(blocks, block)
    return np.sqrt(np.sum(x * x, axis=1, dtype=np.float32), dtype=np.float32)

if __name__ == "__main__":
    V.run(reference)
