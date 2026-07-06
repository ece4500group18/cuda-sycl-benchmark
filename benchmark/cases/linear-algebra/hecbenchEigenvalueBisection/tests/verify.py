#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

LEN = 256
TOL_INTERVAL = 1e-4


def true_eigenvalues():
    d = (2.0 * V.gen_hash01(LEN, 111).astype(np.float64) - 1.0)
    e = 0.5 * (2.0 * V.gen_hash01(LEN - 1, 112).astype(np.float64) - 1.0)
    a = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    return np.sort(np.linalg.eigvalsh(a))


def check(meta, output_path, selftest):
    eig = true_eigenvalues()
    tol = meta["correctness"]["tolerance"]
    if selftest:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        iv = np.empty(2 * LEN)
        iv[0::2] = eig - TOL_INTERVAL / 2
        iv[1::2] = eig + TOL_INTERVAL / 2
        np.savetxt(output_path, iv.astype(V.F32), fmt="%.9g")
    got = V.load_floats(output_path).astype(np.float64)
    if got.size != 2 * LEN:
        return False, "eigenvalue_max_error", float("inf"), tol
    mid = np.sort((got[0::2] + got[1::2]) / 2.0)
    # every converged interval midpoint must sit near a true eigenvalue
    err = float(np.abs(mid - eig).max())
    return err <= tol, "eigenvalue_max_error", err, tol


if __name__ == "__main__":
    V.run_custom(check)
