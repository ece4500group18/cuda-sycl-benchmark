#!/usr/bin/env python3
"""deviceAssertGuard: CPU reference for the per-thread predicate (gtid < N)
recorded by testKernelFlag. Exact 0/1 oracle over all `total` thread ids.

sizes = [total, N] = [1024, 1000] -> first 1000 flags are 1, remaining 24 are 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    total, N = (int(x) for x in meta["input"]["sizes"][:2])
    gtid = np.arange(total, dtype=np.int64)
    return (gtid < N).astype(np.float64)


if __name__ == "__main__":
    V.run(reference)
