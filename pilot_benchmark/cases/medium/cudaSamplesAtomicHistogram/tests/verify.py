#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, bins = meta["input"]["sizes"]
    b = np.floor(V.gen_hash01(n,123)*bins).astype(np.int64)
    return np.bincount(b, minlength=bins).astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
