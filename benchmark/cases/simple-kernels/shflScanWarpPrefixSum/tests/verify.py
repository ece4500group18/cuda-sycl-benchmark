#!/usr/bin/env python3
"""shflScanWarpPrefixSum: CPU reference = segmented inclusive prefix sum of
in[i]=(i%9)+1 with segment length = block size (256, the written config).
Integer addition is associative, so the CPU running sum matches the GPU's
hierarchical warp-shuffle accumulation exactly.

sizes = [n, seg] = [262144, 256]. Output: the 256-wide (multi-warp) scan.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n, seg = (int(x) for x in meta["input"]["sizes"][:2])
    vals = (np.arange(n, dtype=np.int64) % 9) + 1
    return vals.reshape(-1, seg).cumsum(axis=1).reshape(-1).astype(np.float64)


if __name__ == "__main__":
    V.run(reference)
