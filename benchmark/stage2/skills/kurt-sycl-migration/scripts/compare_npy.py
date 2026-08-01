#!/usr/bin/env python3
"""Compare a produced output against a saved golden reference within tolerances.

Usage:
  compare_npy.py got.npy golden.npy [--rtol 1e-5] [--atol 1e-6]

Supports .npy (numpy) and raw float32 .bin (same length inferred by file size).
Exit code 0 = within tolerance, 1 = mismatch, 2 = usage/load error.
"""
import argparse, sys

def load(path):
    if path.endswith(".npy"):
        import numpy as np
        return np.load(path).astype("float64").ravel()
    else:
        import numpy as np
        return np.fromfile(path, dtype="float32").astype("float64").ravel()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("got"); ap.add_argument("golden")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-6)
    a = ap.parse_args()
    try:
        import numpy as np
        got, golden = load(a.got), load(a.golden)
    except Exception as e:
        print(f"load error: {e}", file=sys.stderr); return 2
    if got.shape != golden.shape:
        print(f"shape mismatch: {got.shape} vs {golden.shape}", file=sys.stderr); return 1
    diff = np.abs(got - golden)
    tol = a.atol + a.rtol * np.abs(golden)
    bad = diff > tol
    if bad.any():
        i = int(np.argmax(diff))
        print(f"MISMATCH: {bad.sum()} elems; worst @ {i}: got {got[i]:g} golden {golden[i]:g} "
              f"diff {diff[i]:g}", file=sys.stderr)
        return 1
    print(f"PASS: max_abs={diff.max():g} rtol={a.rtol} atol={a.atol}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
