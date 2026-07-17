#!/usr/bin/env python3
"""Stage 2 CUDA-to-SYCL migration benchmark command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


STAGE2_DIR = Path(__file__).resolve().parent
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from aggregate import aggregate_results  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT, DEFAULT_EXPERIMENT, DEFAULT_MANIFEST, DEFAULT_REPORT_ROOT,
    resolve_repo_path, write_json,
)
from discovery import load_manifest, write_manifest  # noqa: E402
from doctor import probe_environment  # noqa: E402
from preflight import preflight_experiment  # noqa: E402
from runner import load_experiment, plan_experiment, reevaluate_result, run_experiment  # noqa: E402


def _path(value: str | Path) -> Path:
    return resolve_repo_path(value)


def command_manifest(args: argparse.Namespace) -> int:
    path = _path(args.output)
    manifest = write_manifest(path, expected_count=args.expected_count, dataset_id=args.dataset_id)
    print(f"wrote {manifest['case_count']} cases: {path}")
    return 0


def command_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(_path(args.manifest))
    cases = list(manifest["cases"])
    if args.category:
        cases = [record for record in cases if record["category"] == args.category]
    if args.difficulty:
        cases = [record for record in cases if record["difficulty"] == args.difficulty]
    for record in cases:
        print(f"{record['case_id']}\t{record['category']}\t{record['difficulty']}\t{record['path']}")
    print(f"cases={len(cases)} dataset={manifest['dataset_id']} commit={manifest['dataset_commit']}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    result = probe_environment(run_smoke=not args.no_smoke, device_selector=args.device_selector)
    print(json.dumps(result, indent=2))
    if args.output:
        path = _path(args.output)
        write_json(path, result)
        print(f"wrote: {path}")
    return 0 if result["ready_for_intel_gpu_experiments"] else 2


def command_preflight(args: argparse.Namespace) -> int:
    result = preflight_experiment(
        _path(args.experiment), run_smoke=not args.no_smoke, require_intel=not args.skip_intel
    )
    print(json.dumps(result, indent=2))
    if args.output:
        path = _path(args.output)
        write_json(path, result)
        print(f"wrote: {path}")
    return 0 if result["ready"] else 2


def _filters(args: argparse.Namespace) -> dict[str, object]:
    return {
        "case_filters": args.case, "harness_filters": args.harness,
        "model_filters": args.model, "skill_filters": args.skill,
    }


def command_plan(args: argparse.Namespace) -> int:
    experiment_path = _path(args.experiment)
    experiment = load_experiment(experiment_path)
    manifest, runs = plan_experiment(experiment, **_filters(args))
    for planned in runs:
        print(
            f"{planned.run_id}\t{planned.case['category']}\t{planned.case['difficulty']}\t"
            f"adapter={planned.harness['adapter']}"
        )
    counts = Counter(str(run.case["category"]) for run in runs)
    print(
        f"runs={len(runs)} cases={len({run.case['case_id'] for run in runs})} "
        f"harnesses={len({run.harness['slug'] for run in runs})} "
        f"models={len({run.model['slug'] for run in runs})} "
        f"skills={len({run.skill_condition['slug'] for run in runs})} repeats={experiment['repeats']}"
    )
    print(f"dataset={manifest['dataset_id']} commit={manifest['dataset_commit']}")
    print("categories=" + ", ".join(f"{key}:{value}" for key, value in sorted(counts.items())))
    return 0


def command_run(args: argparse.Namespace) -> int:
    experiment_path = _path(args.experiment)
    if not args.skip_preflight:
        gate = preflight_experiment(experiment_path, run_smoke=True, require_intel=True)
        if not gate["ready"]:
            print(json.dumps(gate, indent=2), file=sys.stderr)
            print("ERROR: preflight failed; use --skip-preflight only for controlled diagnostics", file=sys.stderr)
            return 2
    results = run_experiment(
        experiment_path, artifact_root=_path(args.artifact_root), overwrite=args.overwrite,
        **_filters(args),
    )
    failures = 0
    for status, path in results:
        print(f"[{status}] {path}")
        if status not in {"pass", "synthetic", "skipped_existing"}:
            failures += 1
    experiment_id = load_experiment(experiment_path)["experiment_id"]
    summary = aggregate_results(
        experiment_id, artifact_root=_path(args.artifact_root), report_root=_path(args.report_root)
    )
    print(f"runs={len(results)} failures={failures}")
    print(f"report={_path(args.report_root) / experiment_id} results={summary['total_results']}")
    return 1 if failures else 0


def command_aggregate(args: argparse.Namespace) -> int:
    summary = aggregate_results(
        args.experiment_id, artifact_root=_path(args.artifact_root), report_root=_path(args.report_root)
    )
    print(f"results={summary['total_results']} scored={summary['scored_results']} synthetic={summary['synthetic_results']}")
    print(f"report: {_path(args.report_root) / args.experiment_id}")
    return 0


def command_reevaluate(args: argparse.Namespace) -> int:
    experiment_path = _path(args.experiment)
    status, result_path = reevaluate_result(experiment_path, _path(args.result))
    experiment_id = load_experiment(experiment_path)["experiment_id"]
    aggregate_results(
        experiment_id,
        artifact_root=_path(args.artifact_root),
        report_root=_path(args.report_root),
    )
    print(f"[{status}] {result_path}")
    print(f"report: {_path(args.report_root) / experiment_id}")
    return 0 if status == "pass" else 1


def _add_matrix_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", action="append")
    parser.add_argument("--harness", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--skill", action="append")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="freeze CUDA-verified cases into a manifest")
    manifest.add_argument("--output", default=str(DEFAULT_MANIFEST)); manifest.add_argument("--expected-count", type=int, default=250)
    manifest.add_argument("--dataset-id")
    manifest.set_defaults(func=command_manifest)
    listing = sub.add_parser("list", help="list frozen eligible cases")
    listing.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); listing.add_argument("--category"); listing.add_argument("--difficulty")
    listing.set_defaults(func=command_list)
    doctor = sub.add_parser("doctor", help="probe Intel GPU SYCL readiness")
    doctor.add_argument("--no-smoke", action="store_true"); doctor.add_argument("--device-selector", default="level_zero:gpu"); doctor.add_argument("--output")
    doctor.set_defaults(func=command_doctor)
    preflight = sub.add_parser("preflight", help="gate dataset, Intel GPU, harness, model, and skill")
    preflight.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT)); preflight.add_argument("--no-smoke", action="store_true")
    preflight.add_argument("--skip-intel", action="store_true"); preflight.add_argument("--output")
    preflight.set_defaults(func=command_preflight)
    plan = sub.add_parser("plan", help="print the exact experiment matrix")
    plan.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT)); _add_matrix_filters(plan); plan.set_defaults(func=command_plan)
    run = sub.add_parser("run", help="execute experiment cells and automatically refresh the report")
    run.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT)); _add_matrix_filters(run)
    run.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT)); run.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    run.add_argument("--overwrite", action="store_true"); run.add_argument("--skip-preflight", action="store_true")
    run.set_defaults(func=command_run)
    aggregate = sub.add_parser("aggregate", help="aggregate per-run JSON results")
    aggregate.add_argument("--experiment-id", required=True); aggregate.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    aggregate.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT)); aggregate.set_defaults(func=command_aggregate)
    reevaluate = sub.add_parser(
        "reevaluate", help="repeat evaluator build/run/verify without reinvoking the model"
    )
    reevaluate.add_argument("--experiment", required=True)
    reevaluate.add_argument("--result", required=True)
    reevaluate.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    reevaluate.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    reevaluate.set_defaults(func=command_reevaluate)
    return parser


def main() -> int:
    parser = build_parser(); args = parser.parse_args()
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 1


if __name__ == "__main__":
    sys.exit(main())
