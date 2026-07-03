"""Helpers shared by every case's tests/verify.py.

A case verify.py adds the repo ``tools/`` dir to sys.path, imports this
module, defines a ``reference(meta)`` function returning the expected result as
a flat array, and calls ``V.run(reference)``.

Deterministic inputs
--------------------
Two input families are reproduced *bit-for-bit* between the CUDA/SYCL kernels
and this Python reference (all in float32):

1. Simple index formulas (easy cases):
       gen_a[i] = ((i % 17) - 8) * 0.25
       gen_b[i] = ((i % 23) - 11) * 0.5

2. A 32-bit integer hash giving "random-looking" but exactly reproducible
   values in [0,1) (gen_hash01) or [-1,1) (gen_hashsigned). The matching C
   device/host function is:

       __host__ __device__ static inline float h01(unsigned i, unsigned s){
         unsigned h = i*2654435761u + s*2246822519u;
         h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
         return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
       }

   Only the low 24 bits feed the float so the value is exactly representable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

F32 = np.float32
_M = np.uint64(0xFFFFFFFF)


# --- deterministic input generators -----------------------------------------

def gen_a(n):
    i = np.arange(n, dtype=np.int64)
    return (((i % 17) - 8).astype(F32)) * F32(0.25)


def gen_b(n):
    i = np.arange(n, dtype=np.int64)
    return (((i % 23) - 11).astype(F32)) * F32(0.5)


def gen_hash01(n, seed=123):
    """float32 values in [0,1), matching the C h01() helper bit-for-bit."""
    i = np.arange(n, dtype=np.uint64)
    a = (i * np.uint64(2654435761)) & _M
    b = (np.uint64(int(seed) & 0xFFFFFFFF) * np.uint64(2246822519)) & _M
    h = (a + b) & _M
    h = (h ^ (h >> np.uint64(15))) & _M
    h = (h * np.uint64(2246822519)) & _M
    h = (h ^ (h >> np.uint64(13))) & _M
    return ((h & np.uint64(0xFFFFFF)).astype(F32)) / F32(0x1000000)


def gen_hashsigned(n, seed=123):
    """float32 values in [-1,1)."""
    return (gen_hash01(n, seed) * F32(2.0) - F32(1.0)).astype(F32)


def gen_index(n, mod, seed=123):
    """Reproducible non-negative integers in [0, mod), as int64."""
    vals = np.floor(gen_hash01(n, seed) * np.float64(mod)).astype(np.int64)
    return np.clip(vals, 0, mod - 1)


# --- I/O & metadata ----------------------------------------------------------

def load_floats(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().split()
    return np.array([float(x) for x in text], dtype=F32)


def load_metadata():
    main_file = os.path.abspath(sys.argv[0])  # <case>/tests/verify.py
    case_dir = os.path.dirname(os.path.dirname(main_file))
    with open(os.path.join(case_dir, "metadata.json"), encoding="utf-8") as fh:
        return json.load(fh), case_dir


# --- error metrics -----------------------------------------------------------

def max_abs_error(got, ref):
    if got.shape != ref.shape:
        return float("inf")
    return float(np.max(np.abs(got - ref))) if got.size else 0.0


def max_rel_error(got, ref):
    if got.shape != ref.shape:
        return float("inf")
    denom = float(np.max(np.abs(ref))) + 1e-12
    return float(np.max(np.abs(got - ref)) / denom) if got.size else 0.0


def write_verify_result(case_dir, variant, output_path, passed, metric_name, value, tol):
    path = os.path.join(case_dir, "logs", "verify_result.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "case_name": os.path.basename(case_dir),
                "variant": variant,
                "status": "pass" if passed else "fail",
                "metric": metric_name,
                "value": value,
                "tolerance": tol,
                "output": output_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            fh,
            indent=2,
        )
        fh.write("\n")


def finish(passed, metric_name, value, tol):
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict} {metric_name}={value:.3e} tol={tol}")
    sys.exit(0 if passed else 1)


# --- generic driver ----------------------------------------------------------

def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cuda")
    ap.add_argument("--output", required=True)
    # --selftest writes the reference into --output and then verifies it; used
    # to smoke-test the verifier on machines with no CUDA/SYCL toolchain.
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def run(reference):
    """Standard array-comparison driver.

    ``reference(meta)`` returns the expected flat array. Metric and tolerance
    come from metadata.correctness (max_abs_error | max_rel_error | exact).
    """
    args = _parse()
    meta, case_dir = load_metadata()
    ref = np.asarray(reference(meta), dtype=F32).reshape(-1)

    if args.selftest:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.savetxt(args.output, ref, fmt="%.9g")

    got = load_floats(args.output)
    metric = meta["correctness"]["metric"]
    tol = meta["correctness"]["tolerance"] or 0.0
    if metric == "max_rel_error":
        err = max_rel_error(got, ref)
    else:  # max_abs_error or exact (exact uses tol == 0)
        err = max_abs_error(got, ref)
    write_verify_result(case_dir, args.variant, args.output, err <= tol, metric, err, tol)
    finish(err <= tol, metric, err, tol)


def run_custom(check):
    """Driver for non-array checks (residual norm, statistics, ...).

    ``check(meta, output_path, selftest)`` must return (passed, name, value,
    tol). In selftest mode it should also write a known-good output first.
    """
    args = _parse()
    meta, case_dir = load_metadata()
    passed, name, value, tol = check(meta, args.output, args.selftest)
    write_verify_result(case_dir, args.variant, args.output, passed, name, value, tol)
    finish(passed, name, value, tol)
