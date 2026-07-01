#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(2.0) * V.gen_hashsigned(n, 123)
    g = V.F32(1.0) * V.gen_hashsigned(n, 321)
    return x * x

if __name__ == "__main__":
    V.run(reference)
