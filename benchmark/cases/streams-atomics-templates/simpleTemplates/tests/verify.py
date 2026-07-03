#!/usr/bin/env python3
"""simpleTemplates: out must equal in + k (max_abs_error)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.gen_a(n) + np.float32(3.0)


if __name__ == "__main__":
    V.run(reference)
