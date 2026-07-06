#!/usr/bin/env python3
"""binaryPartitionOddEvenReduce: CPU reference for odd count + odd-sum +
even-sum over gen(i) = (i*7+3) % 50, i in [0, size). All three are
order-independent integer accumulations, so the CG kernel's grouped result
must match this exactly.

Output slots: [numOfOdds, sumOfOdds, sumOfEvens]. Metric: integer mismatch
count (tolerance 0).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def expected(size):
    i = np.arange(size, dtype=np.int64)
    g = (i * 7 + 3) % 50
    odd = (g & 1) == 1
    return [int(odd.sum()), int(g[odd].sum()), int(g[~odd].sum())]


def check(meta, output, selftest):
    size = int(meta["input"]["sizes"][0])
    exp = expected(size)
    if selftest:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(v) for v in exp) + "\n")
    with open(output, "r", encoding="utf-8") as fh:
        got = [int(float(tok)) for tok in fh.read().split()]
    mism = 3 if len(got) != 3 else sum(1 for a, b in zip(got, exp) if a != b)
    tol = meta["correctness"]["tolerance"] or 0
    return mism <= tol, "odd_even_mismatch", float(mism), float(tol)


if __name__ == "__main__":
    V.run_custom(check)
