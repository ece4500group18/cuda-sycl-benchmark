#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    H0, W0, H1, W1 = meta["input"]["sizes"]
    x = V.gen_hash01(H0 * W0, 123).reshape(H0, W0)
    y = np.zeros((H1, W1), dtype=np.float32)
    y[:32, :32] = x[:32, ::-1]
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
