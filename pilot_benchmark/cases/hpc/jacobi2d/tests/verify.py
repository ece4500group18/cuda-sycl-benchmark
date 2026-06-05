#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""jacobi2d: K Jacobi iterations, fixed boundary."""


def reference(meta):
    ny, nx, K = meta["input"]["sizes"]
    u = V.gen_hash01(ny * nx, 123).reshape(ny, nx).astype(np.float32).copy()
    for _ in range(K):
        un = u.copy()
        un[1:-1, 1:-1] = np.float32(0.25) * (
            ((u[:-2, 1:-1] + u[2:, 1:-1]) + u[1:-1, :-2]) + u[1:-1, 2:])
        u = un
    return u


if __name__ == "__main__":
    V.run(reference)
