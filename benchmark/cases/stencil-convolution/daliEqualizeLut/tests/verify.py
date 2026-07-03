#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = np.floor(V.F32(256.0) * V.gen_hash01(n,123)).astype(np.int32)
    return (((x * 37 + 13) & 255).astype(np.float32))

if __name__ == "__main__":
    V.run(reference)
