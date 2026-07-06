#!/usr/bin/env python3
"""cgThreadBlockRankSum: each of gridDim.x blocks reduces its threads' ranks
0..blockDim-1, so every block's output is the triangular number
(blockDim-1)*blockDim/2. Exact integer oracle.

sizes = [threadsPerBlock, blocksPerGrid] = [256, 16].
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    tpb, blocks = (int(x) for x in meta["input"]["sizes"][:2])
    val = (tpb - 1) * tpb // 2
    return np.full(blocks, val, dtype=np.float64)


if __name__ == "__main__":
    V.run(reference)
