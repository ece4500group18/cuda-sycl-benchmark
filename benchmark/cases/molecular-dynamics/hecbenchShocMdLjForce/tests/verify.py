#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V


def reference(meta):
    n, k = meta["input"]["sizes"]
    spacing = V.F32(1.25)
    i = np.arange(n)
    ix = (i % 16).astype(V.F32)
    iy = ((i // 16) % 16).astype(V.F32)
    iz = (i // 256).astype(V.F32)
    px = (ix + V.F32(0.5) * V.gen_hash01(n, 1)) * spacing
    py = (iy + V.F32(0.5) * V.gen_hash01(n, 2)) * spacing
    pz = (iz + V.F32(0.5) * V.gen_hash01(n, 3)) * spacing

    lj1, lj2, cutsq = V.F32(1.5), V.F32(2.0), V.F32(6.25)
    fx = np.zeros(n, dtype=V.F32)
    fy = np.zeros(n, dtype=V.F32)
    fz = np.zeros(n, dtype=V.F32)
    # Same j-major accumulation order as the kernel's while loop.
    for j in range(k):
        jidx = V.gen_index(n, n, 100 + j)
        delx = px - px[jidx]
        dely = py - py[jidx]
        delz = pz - pz[jidx]
        r2 = delx * delx + dely * dely + delz * delz
        mask = (r2 > 0) & (r2 < cutsq)
        with np.errstate(divide="ignore"):
            r2inv = np.where(mask, V.F32(1.0) / r2, V.F32(0.0)).astype(V.F32)
        r6inv = r2inv * r2inv * r2inv
        forceC = r2inv * r6inv * (lj1 * r6inv - lj2)
        fx += np.where(mask, delx * forceC, V.F32(0.0)).astype(V.F32)
        fy += np.where(mask, dely * forceC, V.F32(0.0)).astype(V.F32)
        fz += np.where(mask, delz * forceC, V.F32(0.0)).astype(V.F32)

    out = np.empty(3 * n, dtype=V.F32)
    out[0::3] = fx
    out[1::3] = fy
    out[2::3] = fz
    return out


if __name__ == "__main__":
    V.run(reference)
