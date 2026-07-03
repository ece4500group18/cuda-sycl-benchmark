#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

NODES, PICKS = 2000, 3
IN, OUT = 0xFE, 0


def build_graph():
    adj = np.zeros((NODES, NODES), dtype=bool)
    for k in range(PICKS):
        w = V.gen_index(NODES, NODES, 300 + k)
        u = np.arange(NODES)
        sel = w != u
        adj[u[sel], w[sel]] = True
        adj[w[sel], u[sel]] = True
    nidx = np.zeros(NODES + 1, dtype=np.int64)
    nidx[1:] = np.cumsum(adj.sum(axis=1))
    nlist = np.nonzero(adj)[1]
    return nidx, nlist, int(adj.sum())


def dev_hash(val):
    val = np.uint32(val)
    val = np.uint32((np.uint32(val >> np.uint32(16)) ^ val) * np.uint32(0x45D9F3B))
    val = np.uint32((np.uint32(val >> np.uint32(16)) ^ val) * np.uint32(0x45D9F3B))
    return np.uint32(val >> np.uint32(16)) ^ val


def init_priorities(nidx, edges):
    avg = np.float32(edges) / np.float32(NODES)
    scaledavg = np.float32((IN // 2) - 1) * avg
    stat = np.empty(NODES, dtype=np.int32)
    for i in range(NODES):
        degree = int(nidx[i + 1] - nidx[i])
        if degree > 0:
            x = np.float32(degree) - np.float32(dev_hash(i)) * np.float32(2.3283064365386962890625e-10)
            res = int(scaledavg / (avg + x))
            stat[i] = (res + res) | 1
        else:
            stat[i] = IN
    return stat


def reference():
    nidx, nlist, edges = build_graph()
    stat = init_priorities(nidx, edges)
    # Sequential fixed point of the prioritized-selection rule; the result is
    # unique, matching the GPU's lock-free convergence.
    changed = True
    while changed:
        changed = False
        for v in range(NODES):
            nv = stat[v]
            if nv & 1:
                neigh = nlist[nidx[v]:nidx[v + 1]]
                blocked = False
                for u in neigh:
                    su = stat[u]
                    if not (nv > su or (nv == su and v > u)):
                        blocked = True
                        break
                if not blocked:
                    stat[neigh] = OUT
                    stat[v] = IN
                    changed = True
    return stat.astype(V.F32)


def check(meta, output_path, selftest):
    ref = reference()
    if selftest:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savetxt(output_path, ref, fmt="%d")
    got = V.load_floats(output_path)
    tol = meta["correctness"]["tolerance"]
    if got.shape != ref.shape:
        return False, "mis_label_mismatch", 1.0, tol
    # Sanity: every node decided (in or out), and the set is exactly ours.
    err = float(np.mean(got != ref))
    return err <= tol, "mis_label_mismatch", err, tol


if __name__ == "__main__":
    V.run_custom(check)
