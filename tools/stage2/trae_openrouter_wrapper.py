from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    args = parser.parse_args()

    working_dir = Path(args.working_dir).resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = working_dir / args.telemetry_file
    trajectory_path = working_dir / args.trajectory_file

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

    completed = subprocess.run(command, cwd=str(working_dir), check=False)

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

    telemetry_path.write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
