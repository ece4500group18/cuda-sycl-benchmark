#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""gelu: tanh-approx GELU."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    x = (V.gen_hashsigned(n, 123).astype(np.float64)) * 3.0
    k = 0.7978845608028654
    return 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x ** 3)))


if __name__ == "__main__":
    V.run(reference)
