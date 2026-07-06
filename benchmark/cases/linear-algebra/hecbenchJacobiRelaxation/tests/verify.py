#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

N, ITERS = 512, 50


def reference(meta):
    j = np.arange(N, dtype=np.float64)
    edge = np.sin(j * 2 * np.pi / (N - 1)).astype(V.F32)
    f = np.zeros((N, N), dtype=V.F32)   # f[j, i] with IDX(i,j)=i+j*N
    f[:, 0] = edge
    f[:, N - 1] = edge
    f[0, :] = edge
    f[N - 1, :] = edge
    # corners follow the i-branch (i==0 or i==N-1) like upstream
    f[0, 0] = edge[0]; f[N - 1, 0] = edge[N - 1]
    f[0, N - 1] = edge[0]; f[N - 1, N - 1] = edge[N - 1]

    for _ in range(ITERS):
        new = f.copy()
        new[1:-1, 1:-1] = (np.float32(0.25)
                           * ((f[1:-1, 2:] + f[1:-1, :-2]) + f[2:, 1:-1] + f[:-2, 1:-1])
                           ).astype(V.F32)
        f = new
    return f.reshape(-1).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
