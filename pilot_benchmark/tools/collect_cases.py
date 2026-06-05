#!/usr/bin/env python3
"""Inventory and validate every benchmark case.

This does not download anything (the pilot cases are authored in-tree); it
"collects" by walking cases/, validating that each case has the required
files and a metadata.json conforming to the expected schema, and printing a
summary. It exits nonzero if any case is structurally invalid.

Usage:
    python3 tools/collect_cases.py [--category easy] [--case vectorAdd]
"""

from __future__ import annotations

import argparse
import sys

import _common as C

REQUIRED_FILES = [
    "metadata.json",
    "README.md",
    "original/main.cu",
    "original/CMakeLists.txt",
    "original/README.md",
    "tests/verify.py",
]

REQUIRED_DIRS = [
    "original", "syclomatic", "manual_sycl", "input", "output", "logs", "tests",
]

REQUIRED_META_TOP = [
    "case_id", "name", "category", "source", "cuda_features", "libraries",
    "input", "build", "run", "correctness", "syclomatic", "status", "notes",
]

import os


def validate_case(case_dir):
    problems = []
    for rel in REQUIRED_DIRS:
        if not os.path.isdir(os.path.join(case_dir, rel)):
            problems.append(f"missing dir: {rel}/")
    for rel in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(case_dir, rel)):
            problems.append(f"missing file: {rel}")
    try:
        meta = C.load_metadata(case_dir)
    except Exception as exc:
        problems.append(f"metadata.json unreadable: {exc}")
        return problems
    for key in REQUIRED_META_TOP:
        if key not in meta:
            problems.append(f"metadata missing key: {key}")
    if meta.get("case_id") and meta["case_id"] != C.case_id_of(case_dir):
        problems.append(
            f"case_id '{meta['case_id']}' != folder '{C.case_id_of(case_dir)}'"
        )
    if meta.get("category") and meta["category"] != C.category_of(case_dir):
        problems.append(
            f"category '{meta['category']}' != folder '{C.category_of(case_dir)}'"
        )
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category")
    ap.add_argument("--case")
    args = ap.parse_args()

    cases = C.iter_cases(args.category, args.case)
    if not cases:
        print("No cases found.")
        return 0

    ok = 0
    bad = 0
    per_cat = {}
    for case_dir in cases:
        cat = C.category_of(case_dir)
        per_cat.setdefault(cat, 0)
        per_cat[cat] += 1
        problems = validate_case(case_dir)
        cid = C.case_id_of(case_dir)
        if problems:
            bad += 1
            print(f"[INVALID] {cat}/{cid}")
            for p in problems:
                print(f"          - {p}")
        else:
            ok += 1
            print(f"[ok]      {cat}/{cid}")

    print("\n=== collection summary ===")
    for cat in C.CATEGORIES:
        if cat in per_cat:
            print(f"  {cat:12s}: {per_cat[cat]} case(s)")
    print(f"  total       : {len(cases)} case(s)  ({ok} valid, {bad} invalid)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
