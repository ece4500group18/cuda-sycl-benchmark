#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    h, w, q = meta["input"]["sizes"]
    cells = h * w
    f = (V.F32(0.2) + V.F32(0.01) * V.gen_hash01(q * cells, 123)).reshape(q, cells)
    rho = np.sum(f, axis=0, dtype=np.float32)
    ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8]) / rho
    uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8]) / rho
    return rho + V.F32(0.01) * (ux + uy)

if __name__ == "__main__":
    V.run(reference)
