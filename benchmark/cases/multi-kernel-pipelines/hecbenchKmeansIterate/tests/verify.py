#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

NP_, NF, NC, ITERS = 4096, 8, 5, 4


def reference():
    feature = V.gen_hashsigned(NP_ * NF, 17).reshape(NP_, NF)
    clusters = feature[:NC].copy()
    member = np.zeros(NP_, dtype=np.int64)
    for _ in range(ITERS):
        # find_membership: sequential l-loop accumulation in float32
        dist = np.zeros((NP_, NC), dtype=V.F32)
        for l in range(NF):
            d = feature[:, l][:, None] - clusters[:, l][None, :]
            dist += (d * d).astype(V.F32)
        member = np.argmin(dist, axis=1)  # strict <, first index wins
        # host centroid recomputation (float64 accumulation, point order)
        for c in range(NC):
            sel = member == c
            if sel.any():
                clusters[c] = (feature[sel].astype(np.float64).sum(axis=0)
                               / sel.sum()).astype(V.F32)
    return member, clusters


def check(meta, output_path, selftest):
    member, clusters = reference()
    ref = np.concatenate([member.astype(V.F32), clusters.reshape(-1)])
    if selftest:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.savetxt(output_path, ref, fmt="%.9g")
    got = V.load_floats(output_path)
    tol = meta["correctness"]["tolerance"]
    if got.shape != ref.shape:
        return False, "membership_mismatch_fraction", 1.0, tol
    got_member = got[:NP_]
    got_clusters = got[NP_:]
    # Borderline FMA rounding may flip a near-equidistant point's cluster;
    # tolerate a small fraction, but centroids must still agree closely.
    frac = float(np.mean(got_member != member))
    cent_err = float(np.abs(got_clusters - clusters.reshape(-1)).max())
    passed = (frac <= tol) and (cent_err <= 1e-3)
    return passed, "membership_mismatch_fraction", frac, tol


if __name__ == "__main__":
    V.run_custom(check)
