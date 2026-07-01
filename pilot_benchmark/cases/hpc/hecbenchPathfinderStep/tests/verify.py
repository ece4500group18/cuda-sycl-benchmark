#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    rows, cols = meta["input"]["sizes"]
    prev = ((np.arange(cols)*7)%31).astype(np.float32)
    cost = ((np.arange(cols)*13)%17).astype(np.float32)
    y = np.empty(cols, dtype=np.float32)
    for c in range(cols):
        y[c] = cost[c] + min(prev[c], prev[max(c-1,0)], prev[min(c+1,cols-1)])
    return y

if __name__ == "__main__":
    V.run(reference)
