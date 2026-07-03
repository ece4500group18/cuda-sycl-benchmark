#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    return V.F32(0.5) * V.gen_hashsigned(meta["input"]["sizes"][0], 123)

if __name__ == "__main__":
    V.run(reference)
