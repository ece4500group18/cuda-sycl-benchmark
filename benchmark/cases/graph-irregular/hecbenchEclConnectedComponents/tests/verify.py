#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

NODES, PICKS = 2000, 3


def build_edges():
    edges = []
    for k in range(PICKS):
        skip = V.gen_hash01(NODES, 400 + k) < np.float32(0.6)
        w = V.gen_index(NODES, NODES, 500 + k)
        for u in range(NODES):
            if skip[u]:
                continue
            if w[u] != u:
                edges.append((u, int(w[u])))
    return edges


def reference():
    parent = list(range(NODES))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, w in build_edges():
        ru, rw = find(u), find(w)
        if ru != rw:
            parent[max(ru, rw)] = min(ru, rw)
    # ECL-CC hooking always links larger representatives to smaller ones, so
    # the final label of every vertex is its component's minimum vertex id.
    comp_min = {}
    roots = [find(v) for v in range(NODES)]
    for v in range(NODES):
        r = roots[v]
        comp_min[r] = min(comp_min.get(r, v), v)
    return np.array([comp_min[roots[v]] for v in range(NODES)], dtype=np.int64).astype(V.F32)


def check(meta, output_path, selftest):
    ref = reference()
    if selftest:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savetxt(output_path, ref, fmt="%d")
    got = V.load_floats(output_path)
    tol = meta["correctness"]["tolerance"]
    if got.shape != ref.shape:
        return False, "cc_label_mismatch", 1.0, tol
    err = float(np.mean(got != ref))
    return err <= tol, "cc_label_mismatch", err, tol


if __name__ == "__main__":
    V.run_custom(check)
