#!/usr/bin/env python3
"""streamOrderedAllocVectorAdd: CPU reference = c[i] = a[i] + b[i] with
a[i]=(i%23)-11, b[i]=(i%19)-9. Small exact integers, per-element and
order-independent, so the allocation strategy never changes the result.

sizes = [nelem] = [1048576].
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
    return (((i % 23) - 11) + ((i % 19) - 9)).astype(np.float64)


if __name__ == "__main__":
    V.run(reference)
