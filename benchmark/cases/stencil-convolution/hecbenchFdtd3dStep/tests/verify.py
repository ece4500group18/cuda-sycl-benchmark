#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    nx, ny, nz = meta["input"]["sizes"]
    x = V.gen_hashsigned(nx * ny * nz, 123).reshape(nx, ny, nz)
    y = np.empty_like(x)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                y[i, j, k] = x[i, j, k] + V.F32(0.1) * (x[max(i-1,0), j, k] + x[i, max(j-1,0), k] + x[i, j, max(k-1,0)] - V.F32(3.0) * x[i, j, k])
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
