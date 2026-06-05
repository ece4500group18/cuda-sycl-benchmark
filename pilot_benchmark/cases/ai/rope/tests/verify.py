#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""rope: rotary position embedding."""


def reference(meta):
    seq, dim = meta["input"]["sizes"]
    x = V.gen_hash01(seq * dim, 123).reshape(seq, dim).astype(np.float64)
    half = dim // 2
    p = np.arange(seq)[:, None]
    k = np.arange(half)[None, :]
    theta = 10000.0 ** (-2.0 * k / dim)
    ang = p * theta
    cs, sn = np.cos(ang), np.sin(ang)
    x0, x1 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x0 * cs - x1 * sn
    out[:, 1::2] = x0 * sn + x1 * cs
    return out


if __name__ == "__main__":
    V.run(reference)
