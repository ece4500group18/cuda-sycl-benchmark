#!/usr/bin/env python3
"""saxpy: y must equal alpha*x + y (max_abs_error)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n = meta["input"]["sizes"][0]
    return np.float32(2.5) * V.gen_a(n) + V.gen_b(n)


if __name__ == "__main__":
    V.run(reference)
