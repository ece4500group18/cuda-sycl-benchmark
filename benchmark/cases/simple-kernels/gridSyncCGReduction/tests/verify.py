#!/usr/bin/env python3
"""gridSyncCGReduction: CPU reference = plain sum of input[i]=((i%23)-11)*0.5,
i in [0,n). Every value is a multiple of 0.5 with |.|<=5.5, so every partial sum
is exactly representable and the sum is order-independent (exact oracle).

sizes = [n] = [262144]. Output: the single scalar sum (= -32.5).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n = int(meta["input"]["sizes"][0])
    i = np.arange(n, dtype=np.int64)
    v = ((i % 23) - 11).astype(np.float64) * 0.5
    return np.array([v.sum()], dtype=np.float64)


if __name__ == "__main__":
    V.run(reference)
