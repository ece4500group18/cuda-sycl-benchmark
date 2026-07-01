#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H, W = meta["input"]["sizes"]
    x = V.gen_hashsigned(H * W, 123).reshape(H, W)
    y = V.F32(0.5) * x
    y[1:] += V.F32(0.125) * x[:-1]
    y[:-1] += V.F32(0.125) * x[1:]
    y[:, 1:] += V.F32(0.125) * x[:, :-1]
    y[:, :-1] += V.F32(0.125) * x[:, 1:]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
