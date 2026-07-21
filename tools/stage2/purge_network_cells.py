#!/usr/bin/env python3
"""Delete cells whose session ran against an unreachable build worker.

A dropped VPN does not stop a session. The agent writes main.sycl.cpp, then
`sycl_build.sh` fails with `ssh: connect to host ... Connection timed out` and
returncode 255. The runner has no way to tell that apart from a compiler
rejecting the code, so it records funnel="compile_error" -- a scored result,
which means the cell is skipped on resume and the outage stays in the pass rate
forever.

These cells cannot be repaired by re-running build/verify. The agent spends its
whole repair loop reacting to SSH timeouts as though they were compiler
diagnostics, so main.sycl.cpp is whatever the model mangled it into while
chasing errors that never existed, and the token and turn telemetry is inflated
by the same loop. The session is not a measurement of anything. Delete it and
let the case run again.

    python tools/stage2/purge_network_cells.py --experiment <config.json>
    python tools/stage2/purge_network_cells.py --experiment <config.json> --purge

Cells are archived before deletion unless --no-archive is passed. After purging,
refresh the report so the deleted cells leave the KPIs:

    python tools/stage2/cli.py aggregate --experiment-id <experiment_id>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ssh(1) exits 255 on its own connection failures rather than relaying the
# remote command's status. Matching the message as well as the code avoids
# mistaking a compiler that happens to exit 255 for a network problem.
SSH_FAILURE = re.compile(
    r"ssh: connect to host"
    r"|Connection timed out"
    r"|Connection refused"
    r"|No route to host"
    r"|Could not resolve hostname"
    r"|Host key verification failed"
    r"|Connection closed by remote host"
    r"|Broken pipe",
    re.I,
)


def network_failure_in(cell: Path, name: str) -> str | None:
    """Return the offending line if this stage died on the network."""
    path = cell / name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    blob = (data.get("stderr") or "") + "\n" + (data.get("stdout") or "")
    match = SSH_FAILURE.search(blob)
    if not match:
        return None
    for line in blob.splitlines():
        if SSH_FAILURE.search(line):
            return line.strip()
    return match.group(0)


def find_affected(root: Path, cases: list[str] | None = None) -> list[tuple[Path, str, str]]:
    """Return (cell, stage, evidence) for every cell poisoned by a network drop."""
    wanted = set(cases) if cases else None
    affected = []
    for migration in sorted(root.glob("*/*/migration.json")):
        cell = migration.parent
        if wanted is not None and cell.parent.name not in wanted:
            continue
        for stage in ("build.json", "run.json"):
            evidence = network_failure_in(cell, stage)
            if evidence:
                affected.append((cell, stage, evidence))
                break
    return affected


def worker_reachable(target: str, timeout_s: int = 10) -> tuple[bool, str]:
    proc = subprocess.run(
        [
            "ssh",
            "-o", f"ConnectTimeout={timeout_s}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            target,
            "true",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, "reachable"
    return False, (proc.stderr.strip().splitlines() or ["unknown ssh failure"])[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--artifact-root", default=str(REPO_ROOT / "artifacts" / "stage2"))
    parser.add_argument("--purge", action="store_true", help="delete the affected cells")
    parser.add_argument("--no-archive", action="store_true", help="do not keep a copy before deleting")
    parser.add_argument("--archive-root", help="where to put the copy (default: alongside artifacts)")
    parser.add_argument("--check-worker", action="store_true", help="also test that the worker is reachable")
    parser.add_argument("--case", action="append", help="limit to these case ids (repeatable)")
    args = parser.parse_args()

    config_path = Path(args.experiment)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(args.artifact_root) / config["experiment_id"]
    if not root.is_dir():
        print(f"no artifacts at {root}", file=sys.stderr)
        return 2

    total = len(list(root.glob("*/*/migration.json")))
    affected = find_affected(root, args.case)
    scope = f" (case {', '.join(args.case)})" if args.case else ""
    print(f"experiment {config['experiment_id']}{scope}")
    print(f"cells on disk: {total}")
    print(f"cells whose build/run died on the network: {len(affected)}\n")

    for cell, stage, evidence in affected:
        print(f"  {cell.relative_to(root)}")
        print(f"      {stage}: {evidence}")

    if args.check_worker:
        executor = config.get("executor") or {}
        target_env = str(executor.get("target_env") or "STAGE2_SSH_TARGET")
        target = os.environ.get(target_env)
        if not target:
            print(f"\n{target_env} is not set in this shell; skipping the reachability test.")
        else:
            ok, detail = worker_reachable(target)
            print(f"\nworker {target}: {'reachable' if ok else 'UNREACHABLE - ' + detail}")
            if not ok:
                print("reconnect the VPN before rerunning, or the cells will be poisoned again.")

    if not affected:
        return 0
    if not args.purge:
        print(f"\nnothing deleted. rerun with --purge to remove these {len(affected)} cells.")
        return 1

    archive_root = None
    if not args.no_archive:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(args.archive_root) if args.archive_root else Path(args.artifact_root).parent / "stage2_purged"
        archive_root = base / f"{config['experiment_id']}__{stamp}"
        archive_root.mkdir(parents=True, exist_ok=True)

    print()
    for cell, _, _ in affected:
        relative = cell.relative_to(root)
        if archive_root is not None:
            destination = archive_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cell, destination)
        shutil.rmtree(cell)
        # Remove the case directory too if that was its last cell.
        parent = cell.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
        print(f"[purged] {relative}")

    print(f"\npurged {len(affected)} cells")
    if archive_root is not None:
        print(f"archived to {archive_root}")
    print("\nnow refresh the report so they leave the KPIs:")
    print(f"  python tools/stage2/cli.py aggregate --experiment-id {config['experiment_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
