#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

N, ITERS, GRID = 512, 32, 16
RADIUS = np.float32(0.5)


def reference(meta):
    rand_len = ITERS * N
    rX = np.floor(V.gen_hash01(rand_len, 21) * np.float64(100)).astype(V.F32)
    rY = np.floor(V.gen_hash01(rand_len, 22) * np.float64(100)).astype(V.F32)

    pX = np.full(N, np.float32(10.0), dtype=V.F32)
    pY = np.full(N, np.float32(10.0), dtype=V.F32)
    cmap = np.zeros((N, GRID, GRID), dtype=np.int64)

    for it in range(ITERS):
        dispX = rX[it * N:(it + 1) * N] / np.float32(1000.0) - np.float32(0.0495)
        dispY = rY[it * N:(it + 1) * N] / np.float32(1000.0) - np.float32(0.0495)
        pX = pX + dispX
        pY = pY + dispY
        dX = pX - np.trunc(pX)
        dY = pY - np.trunc(pY)
        iX = np.floor(pX).astype(np.int64)
        iY = np.floor(pY).astype(np.int64)
        inside = (pX < GRID) & (pY < GRID) & (pX >= 0) & (pY >= 0)
        hit = inside & (dX * dX + dY * dY <= RADIUS * RADIUS)
        np.add.at(cmap, (np.arange(N)[hit], iY[hit], iX[hit]), 1)

    out = np.empty(2 * N + N * GRID * GRID, dtype=V.F32)
    out[0:2 * N:2] = pX
    out[1:2 * N:2] = pY
    out[2 * N:] = cmap.reshape(-1).astype(V.F32)
    return out


if __name__ == "__main__":
    V.run(reference)
