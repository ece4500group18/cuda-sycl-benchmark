#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    b, h, w, c = meta["input"]["sizes"]
    x = V.gen_hash01(b * h * w * c, meta["input"]["seed"]).reshape(b, h, w, c)
    return x.transpose(0, 3, 1, 2).reshape(-1)


if __name__ == "__main__":
    V.run(reference)
