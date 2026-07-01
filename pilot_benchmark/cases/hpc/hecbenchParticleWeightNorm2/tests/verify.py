#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    w = V.F32(0.001) + V.gen_hash01(n, 123)
    total = V.F32(0.0)
    for i in range(n):
        total += w[i]
    return w / total

if __name__ == "__main__":
    V.run(reference)
