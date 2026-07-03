#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

NF, NB, NSTEPS = 512, 128, 2
REST, SPACING = 1000.0, 0.05
MASS = REST * SPACING ** 3
H, DT = 0.1, 1e-4
ALPHA, SURF_T, C_S = 0.02, 0.01, 10.0


def w_bspline(r):
    C = 1.0 / (np.pi * H ** 3)
    u = r / H
    val = np.where(u < 1.0, 1.0 - 1.5 * u * u + 0.75 * u ** 3,
                   np.where(u < 2.0, 0.25 * np.power(2.0 - u, 3.0), 0.0))
    return np.where(u >= 2.0, 0.0, val * C)


def del_w_bspline(r):
    C = 1.0 / (np.pi * H ** 3)
    u = r / H
    with np.errstate(divide="ignore", invalid="ignore"):
        outer = np.where(r > 0, -3.0 / (4.0 * H * r) * np.power(2.0 - u, 2.0), 0.0)
    val = np.where(u < 1.0, -1.0 / (H * H) * (3.0 - 2.25 * u),
                   np.where(u < 2.0, outer, 0.0))
    return np.where(u >= 2.0, 0.0, val * C)


def reference(meta):
    i = np.arange(NF)
    ix, iy, iz = i % 8, (i // 8) % 8, i // 64
    pos = np.stack([
        0.025 + ix * 0.05 + 0.005 * V.gen_hash01(NF, 1).astype(np.float64),
        0.025 + iy * 0.05 + 0.005 * V.gen_hash01(NF, 2).astype(np.float64),
        0.075 + iz * 0.05 + 0.005 * V.gen_hash01(NF, 3).astype(np.float64),
    ], axis=1)
    v = np.stack([
        0.01 * (2.0 * V.gen_hash01(NF, 4).astype(np.float64) - 1.0),
        0.01 * (2.0 * V.gen_hash01(NF, 5).astype(np.float64) - 1.0),
        0.01 * (2.0 * V.gen_hash01(NF, 6).astype(np.float64) - 1.0),
    ], axis=1)
    v_half = v.copy()
    density = np.full(NF, REST)

    j = np.arange(NB)
    bpos = np.stack([0.025 + (j % 16) * 0.05, 0.025 + (j // 16) * 0.05,
                     np.zeros(NB)], axis=1)
    bn = np.zeros((NB, 3)); bn[:, 2] = 1.0

    eye = np.eye(NF, dtype=bool)
    for _ in range(NSTEPS):
        # --- updatePressures
        diff = pos[:, None, :] - pos[None, :, :]
        r = np.sqrt((diff ** 2).sum(axis=2))
        vdiff = v[:, None, :] - v[None, :, :]
        dw = del_w_bspline(r)
        contrib = (MASS * dw) * (vdiff * diff).sum(axis=2) * DT
        density = density + contrib.sum(axis=1)
        B = REST * C_S * C_S / 7.0
        pressure = B * (np.power(density / REST, 7.0) - 1.0)

        # --- updateAccelerationsFP
        diff = pos[:, None, :] - pos[None, :, :]
        r = np.sqrt((diff ** 2).sum(axis=2))
        dw = del_w_bspline(r)
        wv = w_bspline(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            pterm = (pressure[:, None] / density[:, None] ** 2
                     + pressure[None, :] / density[None, :] ** 2) * MASS * dw
        a_pair = -pterm[:, :, None] * diff

        vdiff = v[:, None, :] - v[None, :, :]
        VdotR = (vdiff * diff).sum(axis=2)
        r2 = (diff ** 2).sum(axis=2)
        eps = H / 10.0
        nu = 2.0 * ALPHA * H * C_S / (density[:, None] + density[None, :])
        stress = nu * VdotR / (r2 + eps * H * H)
        visc = np.where(VdotR < 0.0, MASS * stress * dw, 0.0)
        a_pair += visc[:, :, None] * diff

        a_pair += (SURF_T * wv)[:, :, None] * diff

        a_pair[eye] = 0.0
        a = a_pair.sum(axis=1)
        a[:, 2] += -9.8

        # --- updateAccelerationsBP
        bdiff = pos[:, None, :] - bpos[None, :, :]
        br = np.sqrt((bdiff ** 2).sum(axis=2))
        by = np.sqrt(((bdiff ** 2) * (bn[None, :, :] ** 2)).sum(axis=2))
        bx = br - by
        u = by / H
        xi = np.where((1.0 - bx / H) != 0.0, (bx < H).astype(np.float64), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Cg = xi * 2.0 * 0.02 * C_S * C_S / by
        val = np.where((u > 0.0) & (u < 2.0 / 3.0), 2.0 / 3.0,
              np.where((u < 1.0) & (u > 2.0 / 3.0), 2.0 * u - 1.5 * u * u,
              np.where((u < 2.0) & (u > 1.0), 0.5 * (2.0 - u) ** 2, 0.0)))
        gamma = np.nan_to_num(val * Cg, nan=0.0, posinf=0.0, neginf=0.0)
        a += (gamma[:, :, None] * bn[None, :, :]).sum(axis=1)

        # --- updatePositions (leapfrog)
        v_half = v_half + DT * a
        v = v_half + a * (DT / 2.0)
        pos = pos + DT * v_half

    out = np.empty(6 * NF)
    out[0::6], out[1::6], out[2::6] = pos[:, 0], pos[:, 1], pos[:, 2]
    out[3::6], out[4::6], out[5::6] = v[:, 0], v[:, 1], v[:, 2]
    return out.astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
