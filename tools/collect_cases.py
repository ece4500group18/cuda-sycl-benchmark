#!/usr/bin/env python3
"""Discover and audit Stage 1 CUDA benchmark cases.

This command does not download new code. It walks the repository, finds case
directories with ``metadata.json``, checks the CUDA-ground-truth structure, and
writes ``reports/collection_audit.json``.
"""

from __future__ import annotations

import argparse
import sys

import _stage1_common as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--init-dirs", action="store_true", help="create standard generated directories")
    ap.add_argument("--gitkeep", action="store_true", help="with --init-dirs, add .gitkeep files")
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1

    audits = []
    for case in cases:
        if args.init_dirs:
            C.ensure_case_dirs(case, gitkeep=args.gitkeep)
        audit = C.audit_case(case)
        audits.append(audit)
        missing = len(audit["missing_files"]) + len(audit["missing_dirs"])
        prefix = "[ok]" if not audit["missing_files"] else "[needs]"
        print(f"{prefix:8s} {case.relpath} ({case.case_id}) missing_items={missing}")

    result = {
        "timestamp": C.utc_now(),
        "total_cases": len(audits),
        "by_domain": C.count_by(audits, "domain"),
        "by_difficulty": C.count_by(audits, "difficulty"),
        "by_status": C.count_by(audits, "status"),
        "cases": audits,
    }
    C.write_json(C.REPORTS_DIR / "collection_audit.json", result)
    print("\n=== collection audit ===")
    print(f"total_cases: {len(audits)}")
    print(f"by_domain: {result['by_domain']}")
    print(f"by_difficulty: {result['by_difficulty']}")
    print(f"by_status: {result['by_status']}")
    print(f"wrote: {(C.REPORTS_DIR / 'collection_audit.json').relative_to(C.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
