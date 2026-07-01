#!/usr/bin/env python3
"""Validate case metadata for Stage 1 CUDA dataset use.

The repository currently contains legacy pilot metadata. By default this tool
accepts that compatible schema and reports Stage 1 sidecar metadata status.
Use ``--strict-stage1`` to fail unless each case has strict Stage 1 metadata,
usually generated as ``metadata.stage1.json`` by migrate_metadata_stage1.py.
"""

from __future__ import annotations

import argparse
import sys

import _stage1_common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--strict", action="store_true", help="alias for --strict-stage1")
    ap.add_argument("--strict-stage1", action="store_true", help="require strict Stage 1 metadata sidecars")
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1

    rows = [C.validate_metadata(case) for case in cases]
    compat_failures = [row for row in rows if not row["compat_valid"]]
    strict_failures = [row for row in rows if not row["strict_stage1_valid"]]

    result = {
        "timestamp": C.utc_now(),
        "mode": "strict-stage1" if (args.strict or args.strict_stage1) else "compat",
        "total_cases": len(rows),
        "compat_valid_cases": len(rows) - len(compat_failures),
        "strict_stage1_valid_cases": len(rows) - len(strict_failures),
        "compat_failures": compat_failures,
        "strict_stage1_failures": strict_failures,
        "cases": rows,
    }
    C.write_json(C.REPORTS_DIR / "metadata_validation.json", result)

    for row in rows:
        if row["compat_valid"]:
            status = "strict" if row["strict_stage1_valid"] else "compat"
            suffix = f" warnings={'; '.join(row['warnings'])}" if row["warnings"] else ""
            if not row["strict_stage1_valid"]:
                suffix += f" stage1_missing={'; '.join(row['stage1_problems'])}"
            print(f"[ok:{status}] {row['relpath']}{suffix}")
        else:
            print(f"[FAIL]     {row['relpath']}: {', '.join(row['problems'])}")

    print("\n=== metadata validation ===")
    print(f"total_cases: {len(rows)}")
    print(f"compat_valid_cases: {result['compat_valid_cases']}")
    print(f"strict_stage1_valid_cases: {result['strict_stage1_valid_cases']}")
    print(f"wrote: {(C.REPORTS_DIR / 'metadata_validation.json').relative_to(C.REPO_ROOT)}")
    if compat_failures:
        return 1
    if (args.strict or args.strict_stage1) and strict_failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
