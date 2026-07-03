#!/usr/bin/env python3
"""curandUniform: statistical check (mean ~ 0.5, values in [0,1)).

The exact RNG stream is implementation-specific, so we verify distribution
properties rather than exact values.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def check(meta, output, selftest):
    n = meta["input"]["sizes"][0]
    tol = meta["correctness"]["tolerance"]
    if selftest:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        vals = np.random.default_rng(0).random(n)
        np.savetxt(output, vals, fmt="%.9g")
    vals = V.load_floats(output)
    mean_err = abs(float(vals.mean()) - 0.5)
    in_range = float(vals.min()) >= 0.0 and float(vals.max()) < 1.0 + 1e-6
    passed = (mean_err <= tol) and in_range
    return passed, "abs(mean-0.5)", mean_err, tol


if __name__ == "__main__":
    V.run_custom(check)
