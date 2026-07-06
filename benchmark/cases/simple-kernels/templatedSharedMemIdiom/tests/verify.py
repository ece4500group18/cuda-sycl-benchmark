#!/usr/bin/env python3
"""templatedSharedMemIdiom: CPU reference = out[i] = in[i] * N with in[i]=i,
N=256. Independent per-element multiply (no accumulation); num_threads=256 is a
power of two, so both the int and float instantiations are exact.

sizes = [N] = [256]. Output: 256 ints (i*256) followed by 256 floats (i*256),
i.e. the same values twice, concatenated.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n = int(meta["input"]["sizes"][0])
    v = np.arange(n, dtype=np.int64) * n
    return np.concatenate([v, v]).astype(np.float64)   # int block, then float block


if __name__ == "__main__":
    V.run(reference)
