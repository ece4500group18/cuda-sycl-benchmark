#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""cufftC2C: magnitude spectrum of forward FFT."""


def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hash01(n, 123).astype(np.float64)
    return np.abs(np.fft.fft(x))


if __name__ == "__main__":
    V.run(reference)
