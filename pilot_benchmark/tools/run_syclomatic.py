#!/usr/bin/env python3
"""Run SYCLomatic (c2s) / Intel DPC++ Compatibility Tool (dpct) on each case.

For every case it:
  - reads metadata.json
  - runs the migration tool on original/main.cu, writing output into syclomatic/
  - saves combined stdout/stderr to logs/syclomatic.log
  - updates metadata: syclomatic.{status,command,warnings_count,
    manual_fixes_required} and status.syclomatic_migrate

If neither c2s nor dpct is installed, every case is marked
'skipped_no_syclomatic' (the pipeline keeps going).

Usage:
    python3 tools/run_syclomatic.py [--category easy] [--case vectorAdd]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import _common as C


def count_warnings(output):
    """Count DPCT migration warnings in the tool output.

    SYCLomatic emits lines like 'warning: DPCT1003:...'. We also count the
    distinct DPCTxxxx diagnostic codes as a proxy for manual-fix hot spots.
    """
    warn_lines = len(re.findall(r"\bwarning:", output, flags=re.IGNORECASE))
    dpct_codes = set(re.findall(r"DPCT\d{4}", output))
    return warn_lines, dpct_codes


def migrate_one(case_dir, tool):
    meta = C.load_metadata(case_dir)
    cid = C.case_id_of(case_dir)
    syc = meta.setdefault("syclomatic", {})
    log = C.log_path(case_dir, "syclomatic.log")

    if tool is None:
        syc["status"] = "skipped_no_syclomatic"
        syc["warnings_count"] = None
        syc["manual_fixes_required"] = None
        C.set_status(meta, "syclomatic_migrate", "skipped_no_syclomatic")
        C.save_metadata(case_dir, meta)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("SYCLomatic/dpct not found on PATH; migration skipped.\n")
        print(f"[skip] {cid}: no SYCLomatic (c2s/dpct)")
        return

    out_root = os.path.join(case_dir, "syclomatic")
    os.makedirs(out_root, exist_ok=True)
    # --in-root makes generated #include paths stable; -p-less single-file run.
    command = (
        f"{tool} --out-root syclomatic --in-root original "
        f"original/main.cu --gen-build-script --cuda-include-path=\"$CUDA_INCLUDE_PATH\""
    )
    syc["command"] = command
    rc, out = C.run_logged(command, case_dir, log, timeout=600)

    warn_lines, dpct_codes = count_warnings(out)
    syc["warnings_count"] = warn_lines
    # Any DPCT diagnostic code generally flags a spot a human should review.
    syc["manual_fixes_required"] = bool(dpct_codes)

    produced = [
        f for f in os.listdir(out_root)
        if f.endswith((".cpp", ".dp.cpp", ".h", ".hpp"))
    ] if os.path.isdir(out_root) else []

    if rc == 0 and produced:
        syc["status"] = "done"
        C.set_status(meta, "syclomatic_migrate", "pass")
        print(f"[ok]   {cid}: migrated ({warn_lines} warnings, "
              f"codes={sorted(dpct_codes)})")
    else:
        syc["status"] = "failed"
        C.set_status(meta, "syclomatic_migrate", "fail")
        print(f"[FAIL] {cid}: migration rc={rc}, produced={produced}")

    C.save_metadata(case_dir, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category")
    ap.add_argument("--case")
    args = ap.parse_args()

    tool = C.find_syclomatic()
    if tool is None:
        print("NOTE: SYCLomatic (c2s/dpct) not found; all cases -> "
              "skipped_no_syclomatic")
    else:
        print(f"Using SYCLomatic tool: {tool}")

    cases = C.iter_cases(args.category, args.case)
    for case_dir in cases:
        try:
            migrate_one(case_dir, tool)
        except Exception as exc:  # robust: never let one case abort the batch
            print(f"[ERROR] {C.case_id_of(case_dir)}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
