#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""cudaMemcpyAsyncPinned: out = 3*a."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    return np.float32(3.0) * V.gen_a(n)


if __name__ == "__main__":
    V.run(reference)
