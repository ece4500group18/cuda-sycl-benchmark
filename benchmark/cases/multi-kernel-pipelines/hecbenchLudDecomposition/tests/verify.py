#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

DIM = 64


def input_matrix():
    a = (np.float32(0.5) * V.gen_hashsigned(DIM * DIM, 9)).reshape(DIM, DIM)
    a = a + np.float32(DIM) * np.eye(DIM, dtype=np.float32)
    return a.astype(np.float64)


def cpu_lu(a):
    """Doolittle LU without pivoting (float64), packed like the kernel output."""
    m = a.copy()
    for k in range(DIM - 1):
        m[k + 1:, k] /= m[k, k]
        m[k + 1:, k + 1:] -= np.outer(m[k + 1:, k], m[k, k + 1:])
    return m


def check(meta, output_path, selftest):
    a = input_matrix()
    tol = meta["correctness"]["tolerance"]
    if selftest:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savetxt(output_path, cpu_lu(a).astype(V.F32).reshape(-1), fmt="%.9g")
    got = V.load_floats(output_path)
    if got.size != DIM * DIM:
        return False, "lu_residual", float("inf"), tol
    lu = got.reshape(DIM, DIM).astype(np.float64)
    L = np.tril(lu, -1) + np.eye(DIM)
    U = np.triu(lu)
    # Order-independent oracle: the factorization must reconstruct the input.
    resid = float(np.abs(L @ U - a).max())
    return resid <= tol, "lu_residual", resid, tol


if __name__ == "__main__":
    V.run_custom(check)
