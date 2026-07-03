#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    R, C = meta["input"]["sizes"]
    col = V.gen_hashsigned(R * C, 123).reshape(C, R)
    return col.T.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
