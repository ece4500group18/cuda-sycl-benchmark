#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, bins = meta["input"]["sizes"]
    out = np.zeros(bins, dtype=np.int32)
    for i in range(n):
        out[(i * 17 + 3) & (bins - 1)] += 1
    return out.astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
