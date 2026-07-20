"""Run a case verifier outside the migration sandbox."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from common import read_json
from executor import LocalExecutor


def verify_case_output(
    case_path: Path,
    output_path: Path,
    result_path: Path,
    variant: str,
    timeout_s: float = 300,
    selftest: bool = False,
) -> dict[str, Any]:
    verifier = case_path / "tests" / "verify.py"
    if not verifier.is_file():
        return {"status": "fail", "reason": f"missing verifier: {verifier}"}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(verifier),
        "--variant",
        variant,
        "--output",
        str(output_path),
        "--result-json",
        str(result_path),
    ]
    if selftest:
        command.append("--selftest")
    executed = LocalExecutor().run(command, case_path, timeout_s)
    if result_path.is_file():
        payload = read_json(result_path)
    else:
        payload = {"status": "fail", "reason": "verifier did not write result JSON"}
    payload["command"] = executed.command
    payload["returncode"] = executed.returncode
    payload["elapsed_s"] = executed.elapsed_s
    payload["stdout"] = executed.stdout[-4000:]
    payload["stderr"] = executed.stderr[-4000:]
    if executed.status == "timeout":
        payload["status"] = "timeout"
    elif executed.status != "pass" and payload.get("status") == "pass":
        payload["status"] = "fail"
    return payload
