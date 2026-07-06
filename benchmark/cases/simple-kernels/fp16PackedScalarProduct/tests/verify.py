#!/usr/bin/env python3
"""fp16PackedScalarProduct: CPU reference for the per-block dot-product partials,
using the same block/thread grid-stride decomposition as the GPU kernels. Inputs
a[i]=i%4, b[i]=i%2 (both fp16 lanes equal); every partial sum stays within fp16's
exact-integer range, so a plain integer recompute matches bit-for-bit.

sizes = [size, numBlocks, numThreads] = [262144, 128, 128]. Output: 128 per-block
partials (from the intrinsics kernel); every block totals 4096.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    size, nb, nt = (int(x) for x in meta["input"]["sizes"][:3])
    stride = nb * nt
    out = np.empty(nb, dtype=np.float64)
    for b in range(nb):
        block_sum = 0
        for t in range(nt):
            i = t + nt * b
            ts = 0
            while i < size:
                av = i % 4
                bv = i % 2
                ts += 2 * av * bv   # x lane + y lane (identical formula)
                i += stride
            block_sum += ts
        out[b] = block_sum
    return out


if __name__ == "__main__":
    V.run(reference)
