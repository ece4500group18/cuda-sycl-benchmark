#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    blocks, width = meta["input"]["sizes"]
    x = V.gen_hash01(blocks * width, 123).reshape(blocks, 8, 8)
    out = np.empty((blocks,2), dtype=np.float32)
    cosv = np.cos((2*np.arange(8,dtype=np.float32)+1)*np.float32(np.pi)/V.F32(16.0), dtype=np.float32)
    for b in range(blocks):
        out[b,0] = np.sum(x[b], dtype=np.float32) / V.F32(8.0)
        out[b,1] = V.F32(0.5) * np.sum(x[b] * cosv.reshape(1,8), dtype=np.float32)
    return out.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
