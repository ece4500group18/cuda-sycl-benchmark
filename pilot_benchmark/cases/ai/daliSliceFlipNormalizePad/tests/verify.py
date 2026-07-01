#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    N,H,W,C = meta["input"]["sizes"]
    x = V.gen_hash01(N*H*W*C,123).reshape(N,H,W,C)
    y = np.empty((N,48,48,C), dtype=np.float32)
    for c in range(C):
        y[:,:,:,c] = (x[:,8:56,8:56,c][:,:,::-1] - V.F32(0.1*c)) / V.F32(0.5+0.1*c)
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
