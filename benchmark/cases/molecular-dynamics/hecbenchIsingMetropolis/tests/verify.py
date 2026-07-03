#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

NX, NY_HALF = 128, 64
NSWEEPS = 4
TCRIT = np.float32(2.26918531421)
INV_TEMP = np.float32(1.0) / TCRIT


def randgrid(seed):
    return V.gen_hash01(NX * NY_HALF, seed).reshape(NX, NY_HALF)


def init_spins(seed):
    return np.where(randgrid(seed) < np.float32(0.5), -1, 1).astype(np.int8)


def update(lattice, op_lattice, seed, is_black):
    i = np.arange(NX)[:, None]
    ipp = (i + 1) % NX
    inn = (i - 1) % NX
    j = np.arange(NY_HALF)[None, :]
    jpp = (j + 1) % NY_HALF
    jnn = (j - 1) % NY_HALF
    odd = (i % 2) == 1
    joff = np.where(odd, jpp, jnn) if is_black else np.where(odd, jnn, jpp)

    nn_sum = (op_lattice[inn, j] + op_lattice[i, j] + op_lattice[ipp, j]
              + np.take_along_axis(op_lattice, np.broadcast_to(joff, (NX, NY_HALF)), axis=1))
    lij = lattice
    arg = (np.float32(-2.0) * INV_TEMP) * (nn_sum.astype(np.float32) * lij.astype(np.float32))
    acceptance = np.exp(arg, dtype=np.float32)
    flip = randgrid(seed) < acceptance
    return np.where(flip, -lij, lij).astype(np.int8)


def reference():
    black = init_spins(11)
    white = init_spins(12)
    for s in range(NSWEEPS):
        black = update(black, white, 1000 + 2 * s, True)
        white = update(white, black, 1001 + 2 * s, False)
    return np.concatenate([black.reshape(-1), white.reshape(-1)]).astype(V.F32)


def check(meta, output_path, selftest):
    ref = reference()
    if selftest:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savetxt(output_path, ref, fmt="%d")
    got = V.load_floats(output_path)
    if got.shape != ref.shape:
        return False, "spin_mismatch_fraction", 1.0, meta["correctness"]["tolerance"]
    # Borderline expf-vs-np.exp rounding can legitimately flip a rare spin and
    # cascade a little; tolerate a small site-mismatch fraction.
    frac = float(np.mean(got != ref))
    tol = meta["correctness"]["tolerance"]
    return frac <= tol, "spin_mismatch_fraction", frac, tol


if __name__ == "__main__":
    V.run_custom(check)
