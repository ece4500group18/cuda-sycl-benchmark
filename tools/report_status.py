#!/usr/bin/env python3
"""Generate Stage 1 dataset summary reports."""

from __future__ import annotations

import argparse
import sys

import _stage1_common as C


REAL_SOURCE_TYPES = {"adapted", "copied", "extracted", "simplified"}


def status_list(rows: list[dict], predicate) -> str:
    names = [str(row["case_name"]) for row in rows if predicate(row)]
    return ", ".join(names) if names else "(none)"


def real_project_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if str(row.get("source_type", "")).lower() in REAL_SOURCE_TYPES
    ]


def hand_written_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if str(row.get("source_type", "")).lower() == "authored"
    ]


def extraction_summary(rows: list[dict]) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for row in real_project_rows(rows):
        by_source.setdefault(str(row.get("source") or "unknown"), []).append(row)

    summary: list[dict] = []
    for source, source_rows in sorted(by_source.items()):
        licenses = sorted({str(row.get("license") or "unknown") for row in source_rows})
        methods = sorted({str(row.get("verification_type") or "unknown") for row in source_rows})
        notes = [
            str(row.get("extraction_notes") or "").strip()
            for row in source_rows
            if str(row.get("extraction_notes") or "").strip()
        ]
        summary.append(
            {
                "source_project": source,
                "case_count": len(source_rows),
                "representative_kernels": ", ".join(str(row["case_name"]) for row in source_rows[:8]),
                "license": ", ".join(licenses),
                "extraction_notes": notes[0] if notes else "standalone CUDA extraction/adaptation",
                "verification_method": ", ".join(methods),
                "benchmark_passed": "yes" if all(row.get("actual_perf_status") == "pass" for row in source_rows) else "no",
            }
        )
    return summary


def write_markdown(rows: list[dict], path) -> None:
    total = len(rows)
    by_domain = C.count_by(rows, "domain")
    by_difficulty = C.count_by(rows, "difficulty")
    by_source_project = C.count_by(rows, "source")
    by_license = C.count_by(rows, "license")
    by_declared_status = C.count_by(rows, "declared_status")
    by_actual_status = C.count_by(rows, "actual_status")
    by_actual_verify = C.count_by(rows, "actual_verify_status")
    by_actual_perf = C.count_by(rows, "actual_perf_status")
    summary_rows = [C.summary_row(row) for row in rows]
    real_rows = real_project_rows(rows)
    authored_rows = hand_written_rows(rows)

    lines: list[str] = []
    lines.append("# Stage 1 CUDA Dataset Status")
    lines.append("")
    lines.append(f"Generated: {C.utc_now()}")
    lines.append("")
    lines.append(f"Total cases: **{total}**")
    lines.append("")
    lines.append("## Cases by domain")
    lines.extend(C.as_markdown_table([{"domain": k, "count": v} for k, v in by_domain.items()], ["domain", "count"]))
    lines.append("")
    lines.append("## Cases by difficulty")
    lines.extend(C.as_markdown_table([{"difficulty": k, "count": v} for k, v in by_difficulty.items()], ["difficulty", "count"]))
    lines.append("")
    lines.append("## Cases by source project")
    lines.extend(C.as_markdown_table([{"source_project": k, "count": v} for k, v in by_source_project.items()], ["source_project", "count"]))
    lines.append("")
    lines.append("## Cases by license")
    lines.extend(C.as_markdown_table([{"license": k, "count": v} for k, v in by_license.items()], ["license", "count"]))
    lines.append("")
    lines.append("## Cases by declared status")
    lines.extend(C.as_markdown_table([{"declared_status": k, "count": v} for k, v in by_declared_status.items()], ["declared_status", "count"]))
    lines.append("")
    lines.append("## Cases by actual status")
    lines.extend(C.as_markdown_table([{"actual_status": k, "count": v} for k, v in by_actual_status.items()], ["actual_status", "count"]))
    lines.append("")
    lines.append("## Actual verification status")
    lines.extend(C.as_markdown_table([{"actual_verify_status": k, "count": v} for k, v in by_actual_verify.items()], ["actual_verify_status", "count"]))
    lines.append("")
    lines.append("## Actual performance status")
    lines.extend(C.as_markdown_table([{"actual_perf_status": k, "count": v} for k, v in by_actual_perf.items()], ["actual_perf_status", "count"]))
    lines.append("")
    lines.append("## Build-ready cases")
    lines.append(status_list(rows, lambda row: row["build_ready"]))
    lines.append("")
    lines.append("## Verify-ready cases")
    lines.append(status_list(rows, lambda row: row["verify_ready"]))
    lines.append("")
    lines.append("## Verified on NVIDIA GPU")
    lines.append(status_list(rows, lambda row: row["actual_verify_status"] == "pass"))
    lines.append("")
    lines.append("## NVIDIA performance recorded")
    lines.append(status_list(rows, lambda row: row["actual_perf_status"] == "pass"))
    lines.append("")
    lines.append("## Real-project extracted cases")
    lines.append(status_list(real_rows, lambda row: True))
    lines.append("")
    lines.append("## Hand-written cases")
    lines.append(status_list(authored_rows, lambda row: True))
    lines.append("")
    lines.append("## Real Project Extraction Summary")
    extraction_rows = extraction_summary(rows)
    if extraction_rows:
        lines.extend(
            C.as_markdown_table(
                extraction_rows,
                [
                    "source_project",
                    "case_count",
                    "representative_kernels",
                    "license",
                    "extraction_notes",
                    "verification_method",
                    "benchmark_passed",
                ],
            )
        )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Verify-ready but not actually verified")
    lines.append(status_list(rows, lambda row: row["verify_ready"] and row["actual_verify_status"] != "pass"))
    lines.append("")
    lines.append("## Needs review or rejected")
    review = [
        {
            "case_name": row["case_name"],
            "status": row["status"],
            "notes": row["notes"] or "no explicit review reason recorded",
        }
        for row in rows
        if row["actual_status"] in {"needs_review", "rejected"} or row["failure_reason"]
    ]
    if review:
        lines.extend(C.as_markdown_table(review, ["case_name", "status", "notes"]))
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Dataset summary table")
    lines.extend(C.as_markdown_table(summary_rows, C.SUMMARY_COLUMNS))
    lines.append("")
    lines.append("## How to run Stage 1 on an NVIDIA GPU machine")
    lines.append("")
    lines.append("```bash")
    lines.append("python tools/check_metadata.py")
    lines.append("python tools/migrate_metadata_stage1.py")
    lines.append("python tools/check_metadata.py --strict-stage1")
    lines.append("python tools/collect_cases.py")
    lines.append("python tools/build_case.py --case <case_name>")
    lines.append("python tools/run_case.py --case <case_name>")
    lines.append("python tools/verify_case.py --case <case_name>")
    lines.append("python tools/benchmark_case.py --case <case_name>")
    lines.append("python tools/run_all.py --stage cuda_verify")
    lines.append("python tools/run_all.py --stage cuda_benchmark")
    lines.append("python tools/report_status.py")
    lines.append("```")
    lines.append("")
    lines.append("Stage 2 remains future work: agent migration, migrated-code build/verify, migrated-code runtime, CUDA-vs-migrated performance ratios, token accounting, repair attempts, and end-to-end migration timing are intentionally not evaluated here.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1
    audits = [C.audit_case(case) for case in cases]
    summary_rows = [C.summary_row(row) for row in audits]
    report = {
        "timestamp": C.utc_now(),
        "total_cases": len(audits),
        "by_domain": C.count_by(audits, "domain"),
        "by_difficulty": C.count_by(audits, "difficulty"),
        "by_source_project": C.count_by(audits, "source"),
        "by_license": C.count_by(audits, "license"),
        "by_declared_status": C.count_by(audits, "declared_status"),
        "by_actual_status": C.count_by(audits, "actual_status"),
        "by_actual_verify_status": C.count_by(audits, "actual_verify_status"),
        "by_actual_perf_status": C.count_by(audits, "actual_perf_status"),
        "real_project_extracted_cases": [row["case_name"] for row in real_project_rows(audits)],
        "hand_written_cases": [row["case_name"] for row in hand_written_rows(audits)],
        "real_project_extraction_summary": extraction_summary(audits),
        "cases": audits,
    }
    C.write_json(C.REPORTS_DIR / "dataset_summary.json", report)
    C.write_csv(C.REPORTS_DIR / "dataset_summary.csv", summary_rows, C.SUMMARY_COLUMNS)
    write_markdown(audits, C.REPORTS_DIR / "dataset_summary.md")

    print(f"Total cases: {len(audits)}")
    print(f"By domain: {report['by_domain']}")
    print(f"By difficulty: {report['by_difficulty']}")
    print(f"By source project: {report['by_source_project']}")
    print(f"By license: {report['by_license']}")
    print(f"By declared status: {report['by_declared_status']}")
    print(f"By actual status: {report['by_actual_status']}")
    print(f"Wrote {(C.REPORTS_DIR / 'dataset_summary.md').relative_to(C.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
