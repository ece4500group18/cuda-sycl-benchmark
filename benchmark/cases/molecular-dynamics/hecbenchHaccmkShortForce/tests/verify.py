#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V


def reference(meta):
    n1, n2 = meta["input"]["sizes"]
    xx = V.F32(2.0) * V.gen_hashsigned(n2, 1)
    yy = V.F32(2.0) * V.gen_hashsigned(n2, 2)
    zz = V.F32(2.0) * V.gen_hashsigned(n2, 3)
    mass = V.F32(0.5) + V.gen_hash01(n2, 4)
    vx2 = V.F32(0.1) * V.gen_hashsigned(n1, 5)
    vy2 = V.F32(0.1) * V.gen_hashsigned(n1, 6)
    vz2 = V.F32(0.1) * V.gen_hashsigned(n1, 7)

    fsrmax, mp_rsm, fcoeff = V.F32(0.5), V.F32(0.1), V.F32(0.23)
    ma = [V.F32(x) for x in (0.269327, -0.0750978, 0.0114808,
                             -0.00109313, 0.0000605491, -0.00000147177)]

    xi = np.zeros(n1, dtype=V.F32)
    yi = np.zeros(n1, dtype=V.F32)
    zi = np.zeros(n1, dtype=V.F32)
    # Sequential inner-j accumulation, same order as the kernel.
    for j in range(n2):
        dxc = xx[j] - xx[:n1]
        dyc = yy[j] - yy[:n1]
        dzc = zz[j] - zz[:n1]
        r2 = dxc * dxc + dyc * dyc + dzc * dzc
        m = mass[j] * (r2 < fsrmax).astype(V.F32)
        f = r2 + mp_rsm
        poly = ma[0] + r2 * (ma[1] + r2 * (ma[2] + r2 * (ma[3] + r2 * (ma[4] + r2 * ma[5]))))
        f = m * (V.F32(1.0) / (f * np.sqrt(f)) - poly)
        xi += f * dxc
        yi += f * dyc
        zi += f * dzc

    out = np.empty(3 * n1, dtype=V.F32)
    out[0::3] = vx2 + xi * fcoeff
    out[1::3] = vy2 + yi * fcoeff
    out[2::3] = vz2 + zi * fcoeff
    return out


if __name__ == "__main__":
    V.run(reference)
