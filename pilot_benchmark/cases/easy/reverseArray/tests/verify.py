#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""reverseArray: out[i] = in[n-1-i]."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.gen_a(n)[::-1]


if __name__ == "__main__":
    V.run(reference)
