#!/usr/bin/env python3
"""Print a compact status line for a case, and the campaign total so far.

`cli.py run` prints one `[status] <path>` per cell, which does not say how hard
the session worked or what it cost. This renders the numbers worth watching
while a batch runs.

    python tools/stage2/case_status.py --experiment <config.json> --case vectorAdd
    python tools/stage2/case_status.py --experiment <config.json> --totals

The turn count is deliberately not the `iterations` field. That field records
CodeBuddy's `num_turns`, which counts every message in the transcript -- tool
results included -- and runs about 3x the real turn count, so comparing it to
`budget.max_iterations` suggests the cap is being blown when it is not. What is
counted here is the quantity `--max-turns` actually limits: assistant turns that
issue at least one tool call. A cell at the cap is marked CAP, meaning the
session was cut off rather than finishing on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FUNNEL_ORDER = ["pass", "wrong_output", "run_error", "compile_error", "missing", "harness_error", "synthetic"]


def real_turns(cell: Path) -> int | None:
    """Assistant turns that issued a tool call: what --max-turns caps."""
    stream = cell / "harness_stdout.jsonl"
    if not stream.is_file():
        return None
    ids = set()
    for line in stream.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        blocks = message.get("content") or []
        if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks):
            ids.add(message.get("id"))
    return len(ids)


def human_tokens(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(int(value))


def cells_for(root: Path, case: str | None):
    pattern = f"{case}/*/migration.json" if case else "*/*/migration.json"
    return sorted(root.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--case")
    parser.add_argument("--artifact-root", default=str(REPO_ROOT / "artifacts" / "stage2"))
    parser.add_argument("--totals", action="store_true", help="print the campaign total too")
    parser.add_argument("--indent", default="  ")
    args = parser.parse_args()

    config = json.loads(Path(args.experiment).read_text(encoding="utf-8"))
    root = Path(args.artifact_root) / config["experiment_id"]
    if not root.is_dir():
        print(f"{args.indent}(no artifacts yet)")
        return 0
    cap = int((config.get("budget") or {}).get("max_iterations") or 0)

    if args.case:
        for migration in cells_for(root, args.case):
            data = json.loads(migration.read_text(encoding="utf-8"))
            session = data.get("session") or {}
            skill = str(data.get("skill_condition") or migration.parent.name)
            funnel = str(data.get("funnel"))
            turns = real_turns(migration.parent)
            elapsed = data.get("e2e_elapsed_s")
            if isinstance(turns, int) and cap:
                turn_text = f"{turns}/{cap}" + (" CAP" if turns >= cap else "")
            else:
                turn_text = "-"
            time_text = f"{elapsed:.0f}s" if isinstance(elapsed, (int, float)) else "-"
            flag = "OK " if funnel == "pass" else "   "
            print(
                f"{args.indent}{flag}{skill:<16}{funnel:<14}"
                f"turns {turn_text:<10}"
                f"tok {human_tokens(session.get('tokens_total')):<8}"
                f"{time_text}"
            )

    if args.totals or not args.case:
        counts: dict[str, int] = {}
        total_tokens = 0
        capped = 0
        for migration in cells_for(root, None):
            data = json.loads(migration.read_text(encoding="utf-8"))
            funnel = str(data.get("funnel"))
            counts[funnel] = counts.get(funnel, 0) + 1
            session = data.get("session") or {}
            if isinstance(session.get("tokens_total"), (int, float)):
                total_tokens += session["tokens_total"]
            turns = real_turns(migration.parent)
            if cap and isinstance(turns, int) and turns >= cap:
                capped += 1
        done = sum(counts.values())
        expected = len(config.get("case_ids") or []) * len(config.get("skill_conditions") or [])
        passed = counts.get("pass", 0)
        rate = f"{passed / done * 100:.1f}%" if done else "-"
        ordered = [k for k in FUNNEL_ORDER if k in counts] + [k for k in counts if k not in FUNNEL_ORDER]
        breakdown = " ".join(f"{k}={counts[k]}" for k in ordered)
        print(
            f"{args.indent}campaign: {done}/{expected} cells  "
            f"pass {passed} ({rate})  tok {human_tokens(total_tokens)}  cap-hit {capped}"
        )
        print(f"{args.indent}          {breakdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
