#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n * n * n, 123).reshape(n, n, n)
    y = np.empty_like(x)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                y[i,j,k] = V.F32(0.4)*x[i,j,k] + V.F32(0.1)*(x[max(i-1,0),j,k]+x[min(i+1,n-1),j,k]+x[i,max(j-1,0),k]+x[i,min(j+1,n-1),k]+x[i,j,max(k-1,0)]+x[i,j,min(k+1,n-1)])
    return y.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
