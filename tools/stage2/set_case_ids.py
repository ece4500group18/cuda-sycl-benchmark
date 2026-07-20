#!/usr/bin/env python3
"""Set an experiment's case_ids from the frozen manifest.

The harness templates ship a single smoke case, so an experiment copied from a
template runs one case until its case list is filled in. Editing 250 ids by hand
invites typos that `plan` reports only as a missing case, so do it from the
manifest the experiment already points at.

    # every case in the manifest (the full scored matrix)
    python tools/stage2/set_case_ids.py --experiment <config.json> --all

    # a subset, for a cheaper calibration pass
    python tools/stage2/set_case_ids.py --experiment <config.json> --difficulty hard
    python tools/stage2/set_case_ids.py --experiment <config.json> --category reductions-scans

    # an explicit list, one id per line, '#' comments allowed
    python tools/stage2/set_case_ids.py --experiment <config.json> --file batches/hard50.txt

Filters combine. --dry-run prints the resulting count without writing.

This only widens or narrows what an experiment *may* run; it does not run
anything. Cells already completed under this experiment_id stay completed, so
enlarging a case list is safe and never re-spends tokens on finished work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(experiment: dict) -> list[dict]:
    manifest_path = REPO_ROOT / experiment["dataset_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest["cases"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--all", action="store_true", help="every case in the manifest")
    parser.add_argument("--category", action="append", help="repeatable")
    parser.add_argument("--difficulty", action="append", help="repeatable")
    parser.add_argument("--file", help="text file of case ids, one per line")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.all or args.category or args.difficulty or args.file):
        print("choose at least one of --all / --category / --difficulty / --file", file=sys.stderr)
        return 2

    path = Path(args.experiment)
    experiment = json.loads(path.read_text(encoding="utf-8"))
    cases = load_manifest(experiment)

    selected = {case["case_id"] for case in cases}
    if args.category:
        wanted = set(args.category)
        selected &= {c["case_id"] for c in cases if c.get("category") in wanted}
    if args.difficulty:
        wanted = set(args.difficulty)
        selected &= {c["case_id"] for c in cases if c.get("difficulty") in wanted}
    if args.file:
        listed = {
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        unknown = sorted(listed - {c["case_id"] for c in cases})
        if unknown:
            print(f"not in manifest: {', '.join(unknown)}", file=sys.stderr)
            return 2
        selected &= listed

    if not selected:
        print("no cases matched", file=sys.stderr)
        return 2

    previous = list(experiment.get("case_ids") or [])
    experiment["case_ids"] = sorted(selected)

    dropped = sorted(set(previous) - selected)
    if dropped:
        # aggregate.py rebuilds a report by globbing the artifact directory and
        # never consults the config, so cells already run for a dropped case
        # keep appearing in summary.md. Removing an id here does not remove it
        # from the report.
        print(f"warning: {len(dropped)} case(s) removed from the config but any")
        print("         artifacts already produced for them still enter the report:")
        for case_id in dropped[:10]:
            print(f"           - {case_id}")
        if len(dropped) > 10:
            print(f"           ... and {len(dropped) - 10} more")

    print(f"case_ids: {len(previous)} -> {len(selected)}")
    if args.dry_run:
        print("(dry run, nothing written)")
        return 0

    path.write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
