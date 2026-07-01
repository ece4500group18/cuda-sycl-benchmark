#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    import math
    n = meta["input"]["sizes"][0]
    S = V.F32(10.0) + V.F32(20.0) * V.gen_hash01(n,1)
    K = V.F32(10.0) + V.F32(20.0) * V.gen_hash01(n,2)
    T = V.F32(0.25) + V.F32(2.0) * V.gen_hash01(n,3)
    r = V.F32(0.02); sig = V.F32(0.3)
    sqrtT = np.sqrt(T, dtype=np.float32)
    d1 = (np.log(S/K, dtype=np.float32) + (r + V.F32(0.5)*sig*sig)*T)/(sig*sqrtT)
    d2 = d1 - sig*sqrtT
    erf = np.vectorize(math.erf, otypes=[np.float32])
    cdf1 = V.F32(0.5)*(V.F32(1.0)+erf(d1*V.F32(0.70710678118)).astype(np.float32))
    cdf2 = V.F32(0.5)*(V.F32(1.0)+erf(d2*V.F32(0.70710678118)).astype(np.float32))
    return S*cdf1 - K*np.exp(-r*T, dtype=np.float32)*cdf2

if __name__ == "__main__":
    V.run(reference)
