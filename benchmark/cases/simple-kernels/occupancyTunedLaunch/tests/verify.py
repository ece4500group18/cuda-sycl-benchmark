#!/usr/bin/env python3
"""occupancyTunedLaunch: CPU reference = elementwise square of array[i]=i%1000,
i in [0, arrayCount). Order-independent integer multiply (values <= 998001, no
overflow), so launch configuration never affects the result. Exact oracle.

sizes = [arrayCount] = [1048576].
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n = int(meta["input"]["sizes"][0])
    v = (np.arange(n, dtype=np.int64) % 1000)
    return (v * v).astype(np.float64)


if __name__ == "__main__":
    V.run(reference)
