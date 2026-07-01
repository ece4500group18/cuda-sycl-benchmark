#!/usr/bin/env python3
"""Aggregate per-case performance JSON into CSV and Markdown reports."""

from __future__ import annotations

import csv
import json
import os
import sys

import _common as C

VARIANTS = ["cuda", "sycl"]
COLUMNS = [
    "case_id", "category", "variant", "status", "metric", "median_s",
    "mean_s", "min_s", "max_s", "stdev_s", "repeat", "warmup", "reason",
]


def load_perf(case_dir, variant):
    path = os.path.join(case_dir, "output", f"{variant}_performance.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def row_for(case_dir, variant):
    meta = C.load_metadata(case_dir)
    status = meta.get("status", {}).get(f"{variant}_performance", "unknown")
    data = load_perf(case_dir, variant) or {}
    summary = data.get("summary", {}) or {}
    reason = data.get("reason", "")
    if not reason and status == "skipped_not_built":
        reason = f"{variant}_compile is {meta.get('status', {}).get(f'{variant}_compile', 'unknown')}"
    if not reason and status == "skipped_not_verified":
        reason = f"{variant}_verify is {meta.get('status', {}).get(f'{variant}_verify', 'unknown')}"
    return {
        "case_id": meta.get("case_id", C.case_id_of(case_dir)),
        "category": meta.get("category", C.category_of(case_dir)),
        "variant": variant,
        "status": data.get("status", status),
        "metric": data.get("metric", ""),
        "median_s": summary.get("median", ""),
        "mean_s": summary.get("mean", ""),
        "min_s": summary.get("min", ""),
        "max_s": summary.get("max", ""),
        "stdev_s": summary.get("stdev", ""),
        "repeat": data.get("repeat", ""),
        "warmup": data.get("warmup", ""),
        "reason": reason,
    }


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tally(rows, variant):
    counts = {}
    for row in rows:
        if row["variant"] != variant:
            continue
        key = row["status"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)


def write_md(rows, path):
    lines = ["# Pilot Benchmark Performance\n"]
    lines.append(
        "Metric: end-to-end process runtime in seconds, including program "
        "startup and output writing.\n"
    )

    lines.append("## Summary\n")
    lines.append("| variant | pass | fail | skipped | unknown |")
    lines.append("| --- | --- | --- | --- | --- |")
    for variant in VARIANTS:
        counts = tally(rows, variant)
        nskip = sum(v for k, v in counts.items() if k.startswith("skipped"))
        lines.append(
            f"| {variant} | {counts.get('pass', 0)} | "
            f"{counts.get('fail', 0)} | {nskip} | {counts.get('unknown', 0)} |"
        )
    lines.append("")

    lines.append("## Per-case Detail\n")
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(COLUMNS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[col]) for col in COLUMNS) + " |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    os.makedirs(C.REPORTS_DIR, exist_ok=True)
    rows = [
        row_for(case_dir, variant)
        for case_dir in C.iter_cases()
        for variant in VARIANTS
    ]
    csv_path = os.path.join(C.REPORTS_DIR, "performance_status.csv")
    md_path = os.path.join(C.REPORTS_DIR, "performance_status.md")
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
