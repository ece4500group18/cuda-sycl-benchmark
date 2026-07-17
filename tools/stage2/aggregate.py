"""Aggregate immutable per-run results into KPI JSON, CSV, and Markdown."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from common import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    REPO_ROOT,
    read_json,
    safe_component,
    utc_now,
    write_json,
)


COLUMNS = [
    "experiment_id", "case_id", "category", "difficulty", "harness", "adapter",
    "model", "requested_model_id", "reported_model_id", "skill_condition", "repeat",
    "synthetic", "eligible_for_scoring", "funnel", "migration_success", "tokens_in",
    "cached_input_tokens", "tokens_out", "reasoning_output_tokens", "tokens_total",
    "cost_usd", "cost_source", "migration_wall_clock_s", "e2e_elapsed_s",
    "iterations", "token_budget", "token_budget_exceeded", "session_id", "result_path",
]


def _portable_result_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def collect_results(experiment_id: str, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> list[dict[str, Any]]:
    experiment_id = safe_component(experiment_id, "experiment_id")
    root = artifact_root / experiment_id
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/*/*/repeat-*/migration.json")):
        result = read_json(path)
        session = result.get("session", {})
        rows.append(
            {
                "experiment_id": result.get("experiment_id"), "case_id": result.get("case_id"),
                "category": result.get("category"), "difficulty": result.get("difficulty"),
                "harness": result.get("harness"), "adapter": result.get("adapter"),
                "model": result.get("model"), "requested_model_id": result.get("requested_model_id"),
                "reported_model_id": result.get("reported_model_id"),
                "skill_condition": result.get("skill_condition"), "repeat": result.get("repeat"),
                "synthetic": result.get("synthetic"),
                "eligible_for_scoring": result.get("eligible_for_scoring"),
                "funnel": result.get("funnel"), "migration_success": result.get("migration_success"),
                "tokens_in": session.get("tokens_in"), "tokens_out": session.get("tokens_out"),
                "cached_input_tokens": session.get("cached_input_tokens"),
                "reasoning_output_tokens": session.get("reasoning_output_tokens"),
                "tokens_total": session.get("tokens_total"), "cost_usd": session.get("cost_usd"),
                "cost_source": session.get("cost_source"),
                "migration_wall_clock_s": session.get("wall_clock_s"),
                "e2e_elapsed_s": result.get("e2e_elapsed_s"), "iterations": session.get("iterations"),
                "token_budget": session.get("token_budget"),
                "token_budget_exceeded": session.get("token_budget_exceeded"),
                "session_id": session.get("session_id"),
                "result_path": _portable_result_path(path),
            }
        )
    return rows


def _numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def _group_kpis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["harness"]), str(row["model"]), str(row["skill_condition"]))].append(row)
    result = []
    for (harness, model, skill), group in sorted(grouped.items()):
        scored = [row for row in group if row["eligible_for_scoring"]]
        successes = sum(row["migration_success"] is True for row in scored)
        times = _numbers(scored, "e2e_elapsed_s")
        tokens = _numbers(scored, "tokens_total")
        costs = _numbers(scored, "cost_usd")
        cached = _numbers(scored, "cached_input_tokens")
        reasoning = _numbers(scored, "reasoning_output_tokens")
        iterations = _numbers(scored, "iterations")
        result.append(
            {
                "harness": harness, "model": model, "skill_condition": skill,
                "attempts": len(group), "scored_attempts": len(scored), "successes": successes,
                "pass_rate": successes / len(scored) if scored else None,
                "mean_e2e_s": mean(times) if times else None,
                "median_e2e_s": median(times) if times else None,
                "mean_tokens": mean(tokens) if tokens else None,
                "total_tokens": int(sum(tokens)) if tokens else None,
                "total_cached_input_tokens": int(sum(cached)) if cached else None,
                "total_reasoning_output_tokens": int(sum(reasoning)) if reasoning else None,
                "mean_iterations": mean(iterations) if iterations else None,
                "total_cost_usd": sum(costs) if costs else None,
                "cost_sources": sorted(
                    {str(row["cost_source"]) for row in scored if row.get("cost_source")}
                ),
                "funnel_counts": dict(sorted(Counter(str(row["funnel"]) for row in group).items())),
            }
        )
    return result


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def aggregate_results(
    experiment_id: str,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> dict[str, Any]:
    rows = collect_results(experiment_id, artifact_root)
    scored = [row for row in rows if row["eligible_for_scoring"]]
    kpis = _group_kpis(rows)
    costs_by_source: dict[str, float] = defaultdict(float)
    for row in scored:
        if isinstance(row.get("cost_usd"), (int, float)):
            costs_by_source[str(row.get("cost_source") or "unlabeled")] += float(row["cost_usd"])
    summary = {
        "schema_version": 2, "experiment_id": experiment_id, "generated_at": utc_now(),
        "total_results": len(rows), "scored_results": len(scored),
        "synthetic_results": sum(bool(row["synthetic"]) for row in rows),
        "funnel_counts": dict(sorted(Counter(str(row["funnel"]) for row in rows).items())),
        "scored_successes": sum(row["migration_success"] is True for row in scored),
        "overall_pass_rate": (sum(row["migration_success"] is True for row in scored) / len(scored))
        if scored else None,
        "total_tokens": int(sum(_numbers(scored, "tokens_total"))) if scored else None,
        "cost_usd_by_source": dict(sorted(costs_by_source.items())),
        "kpis_by_harness_model_skill": kpis, "rows": rows,
    }
    destination = report_root / experiment_id
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "summary.json", summary)
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        f"# Stage 2 migration report: {experiment_id}", "", f"Generated: {summary['generated_at']}", "",
        f"- Results: {summary['total_results']} ({summary['scored_results']} scored, "
        f"{summary['synthetic_results']} synthetic)",
        f"- Scored migrations passed: {summary['scored_successes']}",
        f"- Overall pass rate: {_fmt(summary['overall_pass_rate'])}", "",
        f"- Total measured tokens: {_fmt(summary['total_tokens'])}",
        "- Cost USD by source: "
        + (
            ", ".join(f"{key}={value:.6f}" for key, value in summary["cost_usd_by_source"].items())
            or "n/a"
        ),
        "",
        "## Harness x model x skill KPIs", "",
        "| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in kpis:
        markdown.append(
            f"| {row['harness']} | {row['model']} | {row['skill_condition']} | "
            f"{row['scored_attempts']} | {row['successes']} | {_fmt(row['pass_rate'])} | "
            f"{_fmt(row['mean_e2e_s'])} | {_fmt(row['median_e2e_s'])} | "
            f"{_fmt(row['mean_tokens'], 1)} | {_fmt(row['total_cost_usd'], 6)} | "
            f"{', '.join(row['cost_sources']) or 'n/a'} |"
        )
    markdown.extend(["", "## Failure funnel", "", "| status | count |", "| --- | ---: |"])
    markdown.extend(f"| {name} | {count} |" for name, count in summary["funnel_counts"].items())
    markdown.extend(
        ["", "> Synthetic mock results test orchestration only and are excluded from all scored KPIs.", ""]
    )
    (destination / "summary.md").write_text("\n".join(markdown), encoding="utf-8", newline="\n")
    return summary
