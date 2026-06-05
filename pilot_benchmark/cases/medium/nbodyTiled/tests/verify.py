#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""nbodyTiled: gravitational acceleration per body."""


def reference(meta):
    N = meta["input"]["sizes"][0]
    P = np.stack([V.gen_hash01(N, 11), V.gen_hash01(N, 22),
                  V.gen_hash01(N, 33)], axis=1).astype(np.float64)
    d = P[None, :, :] - P[:, None, :]
    dist2 = (d ** 2).sum(axis=2) + 1e-4
    inv3 = dist2 ** -1.5
    acc = (d * inv3[:, :, None]).sum(axis=1)
    return acc.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
