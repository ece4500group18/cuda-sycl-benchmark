#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

N, ITERS = 1024, 5
D = np.float32(0.85)


def reference(meta):
    idx = np.arange(N * N, dtype=np.uint64).reshape(N, N)
    rnd = V.gen_hash01(N * N, 7).reshape(N, N)
    i = np.arange(N)
    pages = ((i[:, None] != i[None, :]) & (rnd < np.float32(0.01)))
    pages[i, (i + 1) % N] = True
    pages = pages.astype(np.float32)
    nout = pages.sum(axis=1).astype(np.float32)

    ranks = np.full(N, np.float32(1.0) / np.float32(N), dtype=V.F32)
    difs = np.zeros(N, dtype=V.F32)
    for _ in range(ITERS):
        maps = pages * (ranks / nout)[:, None]           # map kernel
        new_rank = maps.astype(np.float64).sum(axis=0)    # reduce inner loop
        new_rank = ((np.float32(1.0) - D) / np.float32(N)) + D * new_rank.astype(V.F32)
        difs = np.maximum(np.abs(new_rank - ranks).astype(V.F32), difs)
        ranks = new_rank.astype(V.F32)
    return np.concatenate([ranks, difs]).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
