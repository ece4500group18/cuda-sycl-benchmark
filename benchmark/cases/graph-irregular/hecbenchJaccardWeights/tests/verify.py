#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

NODES, PICKS = 2000, 3
GAMMA = np.float32(0.46)


def build_csr():
    adj = np.zeros((NODES, NODES), dtype=bool)
    for k in range(PICKS):
        w = V.gen_index(NODES, NODES, 300 + k)
        u = np.arange(NODES)
        sel = w != u
        adj[u[sel], w[sel]] = True
        adj[w[sel], u[sel]] = True
    deg = adj.sum(axis=1)
    ptr = np.zeros(NODES + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(deg)
    ind = np.nonzero(adj)[1]
    return ptr, ind, adj


def reference(meta):
    ptr, ind, adj = build_csr()
    e = len(ind)
    rows = np.repeat(np.arange(NODES), np.diff(ptr))
    deg = np.diff(ptr).astype(np.float32)

    # Unweighted: work[row] = degree; Wi = |N(row) ∩ N(col)| (integer-valued
    # float atomics, order-independent); Ws = deg(row)+deg(col).
    inter = (adj @ adj.astype(np.int64))  # pairwise intersection counts
    wi = inter[rows, ind].astype(np.float32)
    ws = (deg[rows] + deg[ind]).astype(np.float32)
    wj = (GAMMA * np.float32(1.0)) * (wi / (ws - wi))
    return wj.astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
