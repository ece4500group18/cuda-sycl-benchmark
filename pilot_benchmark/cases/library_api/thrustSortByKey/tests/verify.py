#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    k = ((np.arange(n, dtype=np.int64) * 17 + 13) % n).astype(np.int32)
    v = np.arange(n, dtype=np.float32)
    return v[np.argsort(k)]

if __name__ == "__main__":
    V.run(reference)
