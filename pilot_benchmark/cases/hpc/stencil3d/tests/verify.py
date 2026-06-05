#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""stencil3d: 7-point clamped stencil."""


def reference(meta):
    nz, ny, nx = meta["input"]["sizes"]
    x = V.gen_hash01(nz * ny * nx, 123).reshape(nz, ny, nx)
    zm = np.clip(np.arange(nz) - 1, 0, nz - 1); zp = np.clip(np.arange(nz) + 1, 0, nz - 1)
    ym = np.clip(np.arange(ny) - 1, 0, ny - 1); yp = np.clip(np.arange(ny) + 1, 0, ny - 1)
    xm = np.clip(np.arange(nx) - 1, 0, nx - 1); xp = np.clip(np.arange(nx) + 1, 0, nx - 1)
    s = x + x[:, :, xm] + x[:, :, xp] + x[:, ym, :] + x[:, yp, :] + x[zm, :, :] + x[zp, :, :]
    return s / np.float32(7.0)


if __name__ == "__main__":
    V.run(reference)
