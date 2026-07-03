#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

SIZE = 64


def system():
    a = (2.0 * V.gen_hash01(SIZE * SIZE, 81).astype(np.float64) - 1.0).reshape(SIZE, SIZE)
    a[np.arange(SIZE), np.arange(SIZE)] += SIZE
    b = 2.0 * V.gen_hash01(SIZE, 82).astype(np.float64) - 1.0
    return a, b


def check(meta, output_path, selftest):
    a, b = system()
    tol = meta["correctness"]["tolerance"]
    if selftest:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savetxt(output_path, np.linalg.solve(a, b).astype(V.F32), fmt="%.9g")
    x = V.load_floats(output_path).astype(np.float64)
    if x.size != SIZE:
        return False, "solve_residual", float("inf"), tol
    # Order-independent oracle: the solution must satisfy the original system.
    resid = float(np.abs(a @ x - b).max())
    return resid <= tol, "solve_residual", resid, tol


if __name__ == "__main__":
    V.run_custom(check)
