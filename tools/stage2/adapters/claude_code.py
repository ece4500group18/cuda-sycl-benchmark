"""Claude Code headless adapter with normalized cost and token telemetry."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from adapters.base import HarnessAdapter, SessionContext, SessionResult
from common import resolve_bash
from prompts import materialize_agent_prompt


def _objects(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _integer(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def parse_stream_json(text: str) -> dict[str, Any]:
    """Parse Claude Code JSONL without depending on one CLI release's shape."""
    events = _objects(text)
    final = next((item for item in reversed(events) if item.get("type") == "result"), {})
    final_usage = final.get("usage") if isinstance(final.get("usage"), dict) else None
    usages: list[dict[str, Any]] = []
    if final_usage:
        usages = [final_usage]
    else:
        for event in events:
            direct = event.get("usage")
            message = event.get("message")
            nested = message.get("usage") if isinstance(message, dict) else None
            usage = direct if isinstance(direct, dict) else nested
            if isinstance(usage, dict):
                usages.append(usage)
    unique = usages
    uncached_input = sum(_integer(item, "input_tokens", "input") for item in unique)
    tokens_out = sum(_integer(item, "output_tokens", "output") for item in unique)
    cache_creation = sum(_integer(item, "cache_creation_input_tokens") for item in unique)
    cache_read = sum(_integer(item, "cache_read_input_tokens") for item in unique)
    tokens_in = uncached_input + cache_creation + cache_read
    reported_model = final.get("model")
    if reported_model is None:
        for event in events:
            message = event.get("message")
            candidate = event.get("model")
            if candidate is None and isinstance(message, dict):
                candidate = message.get("model")
            if candidate:
                reported_model = candidate
                break
    return {
        "events": len(events),
        "status": "completed" if final and not final.get("is_error") else "error",
        "session_id": final.get("session_id"),
        "reported_model": reported_model,
        "iterations": int(final.get("num_turns") or 0),
        "duration_ms": final.get("duration_ms"),
        "duration_api_ms": final.get("duration_api_ms"),
        "cost_usd": final.get("total_cost_usd"),
        "tokens_in": tokens_in if unique else None,
        "tokens_out": tokens_out if unique else None,
        "tokens_total": (tokens_in + tokens_out) if unique else None,
        "cache_creation_input_tokens": cache_creation if unique else None,
        "cache_read_input_tokens": cache_read if unique else None,
        "uncached_input_tokens": uncached_input if unique else None,
        "result": final.get("result"),
    }


class ClaudeCodeAdapter(HarnessAdapter):
    slug = "claude-code"

    def run(self, context: SessionContext) -> SessionResult:
        command_name = str(context.harness.get("command") or "claude")
        command = shutil.which(command_name)
        if not command:
            raise RuntimeError(f"Claude Code executable not found: {command_name}")
        version = context.executor.run([command, "--version"], context.sandbox_path, 30)

        prompt, _ = materialize_agent_prompt(context)

        argv = [
            command,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            context.model_id,
            "--max-turns",
            str(int(context.budget["max_iterations"])),
        ]
        allowed = context.harness.get("allowed_tools")
        if isinstance(allowed, list) and allowed:
            argv.extend(["--allowedTools", *[str(item) for item in allowed]])
        disallowed = context.harness.get("disallowed_tools")
        if isinstance(disallowed, list) and disallowed:
            argv.extend(["--disallowedTools", *[str(item) for item in disallowed]])
        permission_mode = context.harness.get("permission_mode")
        if permission_mode:
            argv.extend(["--permission-mode", str(permission_mode)])

        started = time.perf_counter()
        environment = {"STAGE2_PYTHON": sys.executable}
        bash = resolve_bash()
        if bash:
            environment["PATH"] = (
                str(Path(bash).parent) + os.pathsep + os.environ.get("PATH", "")
            )
        executed = context.executor.run(
            argv,
            context.sandbox_path,
            float(context.budget["wall_clock_s"]),
            env=environment,
        )
        elapsed = time.perf_counter() - started
        (context.run_path / "harness_stdout.jsonl").write_text(
            executed.stdout, encoding="utf-8", newline="\n"
        )
        (context.run_path / "harness_stderr.log").write_text(
            executed.stderr, encoding="utf-8", newline="\n"
        )
        telemetry = parse_stream_json(executed.stdout)
        status = telemetry["status"] if executed.status == "pass" else executed.status
        message = str(telemetry.get("result") or executed.stderr[-1000:] or status)
        return SessionResult(
            status=status,
            tokens_in=telemetry["tokens_in"],
            tokens_out=telemetry["tokens_out"],
            tokens_total=telemetry["tokens_total"],
            wall_clock_s=elapsed,
            iterations=int(telemetry["iterations"]),
            cost_usd=float(telemetry["cost_usd"]) if telemetry["cost_usd"] is not None else None,
            message=message,
            session_id=telemetry["session_id"],
            reported_model=telemetry["reported_model"],
            duration_api_s=(float(telemetry["duration_api_ms"]) / 1000.0)
            if telemetry["duration_api_ms"] is not None
            else None,
            raw_telemetry={
                "event_count": telemetry["events"],
                "cache_creation_input_tokens": telemetry["cache_creation_input_tokens"],
                "cache_read_input_tokens": telemetry["cache_read_input_tokens"],
                "uncached_input_tokens": telemetry["uncached_input_tokens"],
                "process_status": executed.status,
                "returncode": executed.returncode,
                "cli_version": (version.stdout or version.stderr).strip(),
            },
        )


def create_adapter() -> ClaudeCodeAdapter:
    return ClaudeCodeAdapter()
