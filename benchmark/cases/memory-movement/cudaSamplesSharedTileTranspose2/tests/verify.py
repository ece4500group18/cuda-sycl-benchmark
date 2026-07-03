#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H, W = meta["input"]["sizes"]
    return V.gen_hashsigned(H * W, 123).reshape(H, W).T.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
