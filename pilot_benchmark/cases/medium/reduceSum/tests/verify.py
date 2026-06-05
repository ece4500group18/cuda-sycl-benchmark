#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""reduceSum: sum of hashed inputs."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    return np.array([np.float64(V.gen_hash01(n, 123)).sum()], dtype=np.float32)


if __name__ == "__main__":
    V.run(reference)
