#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n, buckets = meta["input"]["sizes"]
    out = np.zeros(buckets, dtype=np.int32)
    for i in range(n):
        b = (i * 17 + 13) & (buckets - 1)
        v = (i * 29 + 7) & 65535
        out[b] = max(out[b], v)
    return out.astype(np.float32)

if __name__ == "__main__":
    V.run(reference)
