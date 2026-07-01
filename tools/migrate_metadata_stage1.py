#!/usr/bin/env python3
"""Generate compatibility-safe Stage 1 metadata sidecars.

This tool writes ``metadata.stage1.json`` next to each legacy ``metadata.json``.
It preserves legacy metadata untouched so older pilot scripts can continue to
use their existing ``status`` object and SYCL-related fields.
"""

from __future__ import annotations

import argparse
import sys

import _stage1_common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--dry-run", action="store_true", help="print actions without writing files")
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1

    failures = 0
    for case in cases:
        doc = C.build_stage1_metadata(case)
        problems = C.stage1_metadata_problems(doc)
        rel = C.stage1_metadata_path(case).relative_to(C.REPO_ROOT)
        if problems:
            failures += 1
            print(f"[needs] {case.relpath}: {', '.join(problems)}")
            continue
        if args.dry_run:
            print(f"[dry]   would write {rel}")
        else:
            C.write_json(C.stage1_metadata_path(case), doc)
            print(f"[ok]    wrote {rel}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
