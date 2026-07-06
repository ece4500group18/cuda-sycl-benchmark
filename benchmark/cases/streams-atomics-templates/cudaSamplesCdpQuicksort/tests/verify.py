#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

N = 4096


def reference(meta):
    vals = np.floor(V.gen_hash01(N, 141).astype(np.float64) * N).astype(np.int64)
    return np.sort(vals).astype(np.float64).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
