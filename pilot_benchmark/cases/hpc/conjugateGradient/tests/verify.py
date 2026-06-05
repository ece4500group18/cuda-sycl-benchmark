#!/usr/bin/env python3
"""conjugateGradient: verify the relative residual ||A x - b|| / ||b||.

A is the diagonally-dominant 1D operator (diag 4, off-diagonals -1).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def apply_A(x):
    y = np.float64(4.0) * x.copy()
    y[1:] -= x[:-1]
    y[:-1] -= x[1:]
    return y


def check(meta, output, selftest):
    N = meta["input"]["sizes"][0]
    tol = meta["correctness"]["tolerance"]
    b = V.gen_hash01(N, 123).astype(np.float64)

    if selftest:
        # Build the dense SPD matrix and solve exactly to produce a good output.
        A = np.zeros((N, N))
        np.fill_diagonal(A, 4.0)
        idx = np.arange(N - 1)
        A[idx, idx + 1] = -1.0
        A[idx + 1, idx] = -1.0
        x = np.linalg.solve(A, b)
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        np.savetxt(output, x, fmt="%.9g")

    x = V.load_floats(output).astype(np.float64)
    res = np.linalg.norm(apply_A(x) - b) / np.linalg.norm(b)
    return res <= tol, "rel_residual", float(res), tol


if __name__ == "__main__":
    V.run_custom(check)
