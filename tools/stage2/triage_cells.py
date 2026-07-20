#!/usr/bin/env python3
"""Triage Stage 2 cells that failed for infrastructure reasons, not model reasons.

A cell whose harness died (quota exhausted, VPN drop, auth expiry) still writes
migration.json with funnel="missing" and eligible_for_scoring=true, so it is
counted as a failed migration and is skipped by a resume run. This script finds
those cells and can delete them so the next `cli.py run` redoes them.

Usage:
    python tools/stage2/triage_cells.py --experiment-id codebuddy-minimax-m3-full250-v1
    python tools/stage2/triage_cells.py --experiment-id ... --purge
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def classify(migration: dict) -> tuple[str, str]:
    """Return (verdict, reason). verdict is one of ok / infra_suspect."""
    funnel = migration.get("funnel")
    session = migration.get("session") or {}
    status = session.get("status")
    telemetry = session.get("raw_telemetry") or {}
    returncode = telemetry.get("returncode")
    tokens = session.get("tokens_total")
    elapsed = migration.get("e2e_elapsed_s")

    if funnel != "missing":
        return "ok", f"funnel={funnel}"
    # funnel == missing: the agent produced no main.sycl.cpp. Decide whether the
    # agent ran and failed the task, or never really ran at all.
    if status != "completed":
        return "infra_suspect", f"session.status={status!r} returncode={returncode}"
    if returncode not in (0, None):
        return "infra_suspect", f"returncode={returncode}"
    if tokens is None:
        return "infra_suspect", "no token telemetry reported"
    if isinstance(elapsed, (int, float)) and elapsed < 15:
        return "infra_suspect", f"suspiciously fast ({elapsed:.1f}s)"
    return "ok", "agent ran to completion but produced no output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--experiment",
        help="experiment JSON; enables cross-checking artifacts against the configured matrix",
    )
    parser.add_argument("--artifact-root", default=str(REPO_ROOT / "artifacts" / "stage2"))
    parser.add_argument(
        "--purge",
        action="store_true",
        help="delete infra_suspect cell directories so a resume run re-executes them",
    )
    args = parser.parse_args()

    config = None
    if args.experiment:
        config = json.loads(Path(args.experiment).read_text(encoding="utf-8"))
        if args.experiment_id and args.experiment_id != config["experiment_id"]:
            print("--experiment-id disagrees with --experiment", file=sys.stderr)
            return 2
        args.experiment_id = config["experiment_id"]
    if not args.experiment_id:
        print("need --experiment-id or --experiment", file=sys.stderr)
        return 2

    root = Path(args.artifact_root) / args.experiment_id
    if not root.is_dir():
        print(f"no artifacts at {root}", file=sys.stderr)
        return 2

    suspects: list[Path] = []
    counts: dict[str, int] = {}
    present: set[tuple[str, str]] = set()
    for migration_path in sorted(root.glob("*/*/migration.json")):
        try:
            migration = json.loads(migration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[unreadable] {migration_path}: {exc}")
            continue
        verdict, reason = classify(migration)
        funnel = str(migration.get("funnel"))
        counts[funnel] = counts.get(funnel, 0) + 1
        present.add((str(migration.get("case_id")), str(migration.get("skill_condition"))))
        if verdict == "infra_suspect":
            suspects.append(migration_path.parent)
            print(f"[infra_suspect] {migration['run_id']}  {reason}")

    total = sum(counts.values())
    print(f"\ncells={total} funnel={dict(sorted(counts.items()))}")
    print(f"infra_suspect={len(suspects)} (counted as failed migrations in the report)")

    if config is not None:
        # aggregate.py globs the artifact directory and never consults the config,
        # so anything extra here silently lands in the report's pass rate.
        expected = {
            (case_id, str(skill["slug"]))
            for case_id in (str(item) for item in config["case_ids"])
            for skill in config["skill_conditions"]
        }
        extra = sorted(present - expected)
        pending = sorted(expected - present)
        print(f"\nconfig expects {len(expected)} cells; artifacts have {len(present)}")
        if extra:
            print(f"EXTRA (in report but not in config, inflates/deflates pass rate): {len(extra)}")
            for case_id, skill in extra:
                print(f"  + {case_id} [{skill}]")
        if pending:
            print(f"pending (not run yet): {len(pending)}")
            for case_id, skill in pending[:10]:
                print(f"  - {case_id} [{skill}]")
            if len(pending) > 10:
                print(f"  ... and {len(pending) - 10} more")
        if not extra and not pending:
            print("artifacts match the configured matrix exactly")

    # harness_error.json cells never wrote migration.json; a resume run retries them.
    orphans = sorted(root.glob("*/*/harness_error.json"))
    if orphans:
        print(f"harness_error cells={len(orphans)} (these are retried automatically)")

    if not suspects:
        return 0
    if not args.purge:
        print("\nrerun with --purge to delete these cells, then rerun cli.py run to redo them")
        return 1
    for cell in suspects:
        shutil.rmtree(cell)
        print(f"[purged] {cell.relative_to(root)}")
    print(f"\npurged {len(suspects)} cells; rerun cli.py run to re-execute them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
