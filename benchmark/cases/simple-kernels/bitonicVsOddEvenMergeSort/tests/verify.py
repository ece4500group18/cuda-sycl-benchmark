#!/usr/bin/env python3
"""bitonicVsOddEvenMergeSort: CPU reference = ascending sort-by-key of the
deterministic (key, val) array. Keys are distinct (40503 invertible mod 65536),
so the sorted permutation is unambiguous and this is an exact integer oracle.

Inputs: key[i] = (i*40503) % 65536, val[i] = i, for i in [0, n).
Output (from the bitonic kernel): "key val" per line -> flat interleaved
[k0,v0,k1,v1,...] of length 2n.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n = int(meta["input"]["sizes"][0])   # 1024
    i = np.arange(n, dtype=np.int64)
    key = ((i * 40503) % 65536).astype(np.int64)
    val = i.copy()
    order = np.argsort(key, kind="stable")   # keys distinct
    ks = key[order]
    vs = val[order]
    out = np.empty(2 * n, dtype=np.float64)
    out[0::2] = ks
    out[1::2] = vs
    return out


if __name__ == "__main__":
    V.run(reference)
