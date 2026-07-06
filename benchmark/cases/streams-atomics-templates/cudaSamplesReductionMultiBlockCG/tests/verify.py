#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

N = 1 << 20


def reference(meta):
    vals = (np.float32(2.0) * V.gen_hash01(N, 131) - np.float32(1.0)).astype(np.float64)
    return np.array([vals.sum()], dtype=np.float64).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
