#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x1 = V.gen_hash01(n, 11) * V.F32(0.5)
    y1 = V.gen_hash01(n, 22) * V.F32(0.5)
    w = V.F32(0.1) + V.F32(0.4) * V.gen_hash01(n, 33)
    h = V.F32(0.1) + V.F32(0.4) * V.gen_hash01(n, 44)
    out = np.empty((n, 4), dtype=np.float32)
    out[:,0] = x1 + V.F32(0.5) * w
    out[:,1] = y1 + V.F32(0.5) * h
    out[:,2] = np.log(w).astype(np.float32)
    out[:,3] = np.log(h).astype(np.float32)
    return out.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
