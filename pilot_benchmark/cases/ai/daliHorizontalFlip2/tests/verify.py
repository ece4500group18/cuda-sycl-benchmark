#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(N, H, W, C)
    return x[:, :, ::-1, :].reshape(-1)

if __name__ == "__main__":
    V.run(reference)
