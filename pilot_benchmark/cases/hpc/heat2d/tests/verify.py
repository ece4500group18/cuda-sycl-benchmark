#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""heat2d: K explicit FTCS steps, fixed boundary."""


def reference(meta):
    ny, nx, K = meta["input"]["sizes"]
    a = np.float32(0.2)
    u = V.gen_hash01(ny * nx, 123).reshape(ny, nx).astype(np.float32).copy()
    for _ in range(K):
        un = u.copy()
        lap = (((u[:-2, 1:-1] + u[2:, 1:-1]) + u[1:-1, :-2]) + u[1:-1, 2:]) - np.float32(4.0) * u[1:-1, 1:-1]
        un[1:-1, 1:-1] = u[1:-1, 1:-1] + a * lap
        u = un
    return u


if __name__ == "__main__":
    V.run(reference)
