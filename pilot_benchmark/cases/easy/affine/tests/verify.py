#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""affine: out = 2*a + 1."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    return np.float32(2.0) * V.gen_a(n) + np.float32(1.0)


if __name__ == "__main__":
    V.run(reference)
