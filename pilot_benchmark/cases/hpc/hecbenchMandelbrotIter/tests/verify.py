#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    W, H, maxit = meta["input"]["sizes"]
    out = np.zeros(W * H, dtype=np.float32)
    for idx in range(W * H):
        px = idx % W; py = idx // W
        cr = V.F32(-2.0) + V.F32(3.0) * V.F32(px) / V.F32(W - 1)
        ci = V.F32(-1.5) + V.F32(3.0) * V.F32(py) / V.F32(H - 1)
        zr = V.F32(0.0); zi = V.F32(0.0); it = 0
        while it < maxit and zr * zr + zi * zi <= V.F32(4.0):
            nzr = zr * zr - zi * zi + cr
            zi = V.F32(2.0) * zr * zi + ci
            zr = nzr; it += 1
        out[idx] = V.F32(it)
    return out

if __name__ == "__main__":
    V.run(reference)
