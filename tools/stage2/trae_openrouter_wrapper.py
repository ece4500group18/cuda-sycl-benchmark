from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_usage(interactions: list[dict[str, Any]], key: str) -> int:
    total = 0
    found = False
    for item in interactions:
        usage = ((item or {}).get("response") or {}).get("usage") or {}
        value = usage.get(key)
        if isinstance(value, int):
            total += value
            found = True
    return total if found else 0


def _trajectory_to_telemetry(trajectory: dict[str, Any], fallback_model: str) -> dict[str, Any]:
    interactions = trajectory.get("llm_interactions")
    if not isinstance(interactions, list):
        interactions = []
    input_tokens = _sum_usage(interactions, "input_tokens")
    output_tokens = _sum_usage(interactions, "output_tokens")
    cached_input_tokens = _sum_usage(interactions, "cache_read_input_tokens")
    reasoning_output_tokens = _sum_usage(interactions, "reasoning_tokens")
    reported_model = trajectory.get("model") or fallback_model
    final_result = trajectory.get("final_result")
    execution_time = trajectory.get("execution_time")
    message = None
    if isinstance(final_result, str) and final_result.strip():
        message = final_result.strip()
    elif isinstance(execution_time, (int, float)):
        message = f"Parsed from Trae trajectory; execution_time_s={execution_time:.3f}"
    else:
        message = "Parsed from Trae trajectory"
    return {
        "tokens_in": input_tokens,
        "tokens_out": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "tokens_total": input_tokens + output_tokens,
        "iterations": len(interactions),
        "cost_usd": None,
        "session_id": None,
        "model": reported_model,
        "message": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trae-exe", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--telemetry-file", default="stage2_telemetry.json")
    parser.add_argument("--trajectory-file", default="stage2_trajectory.json")
    parser.add_argument("--stdout-log", default="trae_cli_stdout.log")
    parser.add_argument("--stderr-log", default="trae_cli_stderr.log")
    args = parser.parse_args()

    working_dir = Path(args.working_dir).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = working_dir / args.telemetry_file
    trajectory_path = working_dir / args.trajectory_file
    stdout_log_path = working_dir / args.stdout_log
    stderr_log_path = working_dir / args.stderr_log

    command = [
        str(Path(args.trae_exe).resolve()),
        "run",
        "--config-file",
        str(Path(args.config_file).resolve()),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--working-dir",
        str(working_dir),
        "--file",
        str(Path(args.prompt_file).resolve()),
        "--max-steps",
        str(args.max_steps),
        "--trajectory-file",
        str(trajectory_path),
    ]

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TERM"] = env.get("TERM", "dumb")
    completed = subprocess.run(
        command,
        cwd=str(working_dir),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    stdout_log_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log_path.write_text(completed.stderr or "", encoding="utf-8")

    telemetry: dict[str, Any]
    if trajectory_path.is_file():
        try:
            trajectory = _read_json(trajectory_path)
            telemetry = _trajectory_to_telemetry(trajectory, args.model)
        except Exception as exc:
            telemetry = {
                "tokens_in": None,
                "tokens_out": None,
                "cached_input_tokens": None,
                "reasoning_output_tokens": None,
                "tokens_total": None,
                "iterations": None,
                "cost_usd": None,
                "session_id": None,
                "model": args.model,
                "message": f"Failed to parse trajectory: {exc}",
            }
    else:
        telemetry = {
            "tokens_in": None,
            "tokens_out": None,
            "cached_input_tokens": None,
            "reasoning_output_tokens": None,
            "tokens_total": None,
            "iterations": None,
            "cost_usd": None,
            "session_id": None,
            "model": args.model,
            "message": "Trae CLI did not produce a trajectory file",
        }

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()
        if tail:
            telemetry["message"] = tail[-1000:]

    telemetry_path.write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "returncode": completed.returncode,
        "stdout_log": str(stdout_log_path),
        "stderr_log": str(stderr_log_path),
        "trajectory_file": str(trajectory_path),
        "telemetry_file": str(telemetry_path),
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
