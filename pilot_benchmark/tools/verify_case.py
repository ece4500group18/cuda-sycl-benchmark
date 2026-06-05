#!/usr/bin/env python3
"""Verify a case's output against its reference using the case's verify.py.

Each case ships tests/verify.py which:
  - regenerates the deterministic inputs (or reads input/)
  - computes a CPU reference
  - compares against output/<variant>_output.txt with a tolerance
  - prints PASS or FAIL and exits 0 (pass) / nonzero (fail)

This orchestrator runs that script per variant and records status.<v>_verify.
Verification is skipped (mirroring the run status) when the variant did not
produce an output file.

Usage:
    python3 tools/verify_case.py --variant cuda [--category easy] [--case ...]
    python3 tools/verify_case.py --variant sycl
"""

from __future__ import annotations

import argparse
import os
import sys

import _common as C


def verify_one(case_dir, variant):
    meta = C.load_metadata(case_dir)
    cid = C.case_id_of(case_dir)
    status = meta.get("status", {})
    run_key = f"{variant}_run"
    verify_key = f"{variant}_verify"
    out_file = os.path.join("output", f"{variant}_output.txt")
    abs_out = os.path.join(case_dir, out_file)
    log = C.log_path(case_dir, "verify.log")

    run_state = status.get(run_key, "unknown")
    if run_state != "pass" or not os.path.isfile(abs_out):
        # Mirror the reason the run did not happen so the report is informative.
        if run_state.startswith("skipped"):
            mirror = run_state.replace("_run", "_verify") if "_run" in run_state else run_state
            C.set_status(meta, verify_key, run_state)
        else:
            C.set_status(meta, verify_key, "skipped_no_output")
        C.save_metadata(case_dir, meta)
        print(f"[skip] {cid}: {variant} verify ({status.get(verify_key)})")
        return

    command = (
        f"python3 tests/verify.py --variant {variant} --output {out_file}"
    )
    rc, out = C.run_logged(command, case_dir, log, timeout=300)
    passed = (rc == 0) and ("PASS" in out)
    C.set_status(meta, verify_key, "pass" if passed else "fail")
    C.save_metadata(case_dir, meta)
    print(f"[{'ok' if passed else 'FAIL'}]   {cid}: {variant} verify")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cuda", "sycl"], required=True)
    ap.add_argument("--category")
    ap.add_argument("--case")
    args = ap.parse_args()

    for case_dir in C.iter_cases(args.category, args.case):
        try:
            verify_one(case_dir, args.variant)
        except Exception as exc:
            print(f"[ERROR] {C.case_id_of(case_dir)}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
