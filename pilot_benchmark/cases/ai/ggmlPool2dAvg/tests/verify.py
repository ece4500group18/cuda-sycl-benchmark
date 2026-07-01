#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, c, h, w = meta["input"]["sizes"]
    x = V.gen_hashsigned(n * c * h * w, 123).reshape(n, c, h, w)
    y = V.F32(0.25) * (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2])
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
