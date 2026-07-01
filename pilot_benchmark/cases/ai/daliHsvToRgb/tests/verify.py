#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    h = V.gen_hash01(n,11); s = V.gen_hash01(n,22); v = V.gen_hash01(n,33)
    out = np.empty((n,3), dtype=np.float32)
    for i in range(n):
        hh = float(h[i] * V.F32(6.0)); k = int(np.floor(hh)); f = V.F32(hh-k)
        p = v[i]*(V.F32(1.0)-s[i]); q = v[i]*(V.F32(1.0)-s[i]*f); t = v[i]*(V.F32(1.0)-s[i]*(V.F32(1.0)-f))
        m = k % 6
        if m == 0: out[i] = [v[i], t, p]
        elif m == 1: out[i] = [q, v[i], p]
        elif m == 2: out[i] = [p, v[i], t]
        elif m == 3: out[i] = [p, q, v[i]]
        elif m == 4: out[i] = [t, p, v[i]]
        else: out[i] = [v[i], p, q]
    return out.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
