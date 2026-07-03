#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""stencil2d: 5-point clamped stencil."""


def reference(meta):
    ny, nx = meta["input"]["sizes"]
    x = V.gen_hash01(ny * nx, 123).reshape(ny, nx)
    up = np.clip(np.arange(ny) - 1, 0, ny - 1); dn = np.clip(np.arange(ny) + 1, 0, ny - 1)
    lf = np.clip(np.arange(nx) - 1, 0, nx - 1); rt = np.clip(np.arange(nx) + 1, 0, nx - 1)
    return np.float32(0.2) * (x + x[up] + x[dn] + x[:, lf] + x[:, rt])


if __name__ == "__main__":
    V.run(reference)
