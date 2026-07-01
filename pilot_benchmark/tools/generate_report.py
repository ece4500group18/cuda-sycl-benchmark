#!/usr/bin/env python3
"""Aggregate every case's metadata.json into reports/pilot_status.{csv,md}.

One row per case with the columns required by the project spec, plus a summary
section in the Markdown report counting how many cases reached each pipeline
stage.

Usage:
    python3 tools/generate_report.py
"""

from __future__ import annotations

import csv
import os
import sys

import _common as C

COLUMNS = [
    "case_id", "category", "name", "cuda_features", "libraries",
    "cuda_compile", "cuda_run", "cuda_verify", "cuda_performance",
    "syclomatic_migrate", "sycl_compile", "sycl_run", "sycl_verify",
    "sycl_performance",
    "warnings_count", "manual_fixes_required", "notes",
]

# Pipeline stages we summarise pass-counts for.
STAGE_KEYS = [
    "cuda_compile", "cuda_run", "cuda_verify", "cuda_performance",
    "syclomatic_migrate", "sycl_compile", "sycl_run", "sycl_verify",
    "sycl_performance",
]


def row_for(case_dir):
    meta = C.load_metadata(case_dir)
    st = meta.get("status", {})
    syc = meta.get("syclomatic", {})
    return {
        "case_id": meta.get("case_id", C.case_id_of(case_dir)),
        "category": meta.get("category", C.category_of(case_dir)),
        "name": meta.get("name", ""),
        "cuda_features": ";".join(meta.get("cuda_features", [])),
        "libraries": ";".join(meta.get("libraries", [])),
        "cuda_compile": st.get("cuda_compile", "unknown"),
        "cuda_run": st.get("cuda_run", "unknown"),
        "cuda_verify": st.get("cuda_verify", "unknown"),
        "cuda_performance": st.get("cuda_performance", "unknown"),
        "syclomatic_migrate": st.get("syclomatic_migrate", "unknown"),
        "sycl_compile": st.get("sycl_compile", "unknown"),
        "sycl_run": st.get("sycl_run", "unknown"),
        "sycl_verify": st.get("sycl_verify", "unknown"),
        "sycl_performance": st.get("sycl_performance", "unknown"),
        "warnings_count": syc.get("warnings_count"),
        "manual_fixes_required": syc.get("manual_fixes_required"),
        "notes": meta.get("notes", ""),
    }


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def tally(rows, key):
    counts = {}
    for r in rows:
        counts[r[key]] = counts.get(r[key], 0) + 1
    return counts


def write_md(rows, path):
    total = len(rows)
    lines = []
    lines.append("# Pilot Benchmark Status\n")
    lines.append(f"Total cases: **{total}**\n")

    # Per-category counts.
    by_cat = tally(rows, "category")
    lines.append("## Cases by category\n")
    lines.append("| category | count |")
    lines.append("| --- | --- |")
    for cat in C.CATEGORIES:
        if cat in by_cat:
            lines.append(f"| {cat} | {by_cat[cat]} |")
    lines.append("")

    # Stage pass-counts.
    lines.append("## Pipeline stage summary\n")
    lines.append("| stage | pass | fail | skipped | unknown |")
    lines.append("| --- | --- | --- | --- | --- |")
    for stage in STAGE_KEYS:
        counts = tally(rows, stage)
        npass = counts.get("pass", 0)
        nfail = counts.get("fail", 0)
        nskip = sum(v for k, v in counts.items() if k.startswith("skipped"))
        nunknown = counts.get("unknown", 0)
        lines.append(f"| {stage} | {npass} | {nfail} | {nskip} | {nunknown} |")
    lines.append("")

    # Full table.
    lines.append("## Per-case detail\n")
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = []
        for c in COLUMNS:
            v = r[c]
            v = "" if v is None else str(v)
            cells.append(v.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    rows = [row_for(cd) for cd in C.iter_cases()]
    csv_path = os.path.join(C.REPORTS_DIR, "pilot_status.csv")
    md_path = os.path.join(C.REPORTS_DIR, "pilot_status.md")
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Total cases: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
