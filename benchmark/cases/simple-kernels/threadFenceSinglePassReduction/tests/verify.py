#!/usr/bin/env python3
"""threadFenceSinglePassReduction: CPU reference for the two reduction results.

input[i] = ((i%29)-14)*0.25 -- every value is a multiple of 0.25 with bounded
partial sums, so every float32 addition anywhere in either reduction tree is
exact (no rounding) regardless of grouping/order. Both the single-pass and the
two-kernel-launch reductions therefore equal the unique exact total (-21.0).

sizes = [N] = [131072]. Output: two lines [single_pass_sum, two_launch_sum],
both equal to the exact total.
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
    total = (((i % 29) - 14).astype(np.float64) * 0.25).sum()   # exact (multiples of 0.25)
    return np.array([total, total], dtype=np.float64)


if __name__ == "__main__":
    V.run(reference)
