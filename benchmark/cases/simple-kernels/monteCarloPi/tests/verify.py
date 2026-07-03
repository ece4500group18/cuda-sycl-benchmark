#!/usr/bin/env python3
"""monteCarloPi: verify the pi estimate is within tolerance of math.pi."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def check(meta, output, selftest):
    tol = meta["correctness"]["tolerance"]
    if selftest:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        np.savetxt(output, np.array([math.pi]), fmt="%.9g")
    val = float(V.load_floats(output)[0])
    err = abs(val - math.pi)
    return err <= tol, "abs_error_to_pi", err, tol


if __name__ == "__main__":
    V.run_custom(check)
