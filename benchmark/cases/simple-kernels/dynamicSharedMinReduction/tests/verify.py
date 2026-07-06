#!/usr/bin/env python3
"""dynamicSharedMinReduction: CPU reference = minimum of input[0..n),
input[i] = ((i % 37) - 18) * 0.5. Every block reads from offset 0, so all
NUM_BLOCKS outputs equal that single minimum. min is order-free and every value
is an exact float, so this is an exact oracle.

sizes = [n, NUM_BLOCKS] = [128, 8].
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n, blocks = (int(x) for x in meta["input"]["sizes"][:2])
    i = np.arange(n, dtype=np.int64)
    vals = (((i % 37) - 18).astype(np.float32)) * np.float32(0.5)
    return np.full(blocks, np.min(vals), dtype=np.float32)


if __name__ == "__main__":
    V.run(reference)
