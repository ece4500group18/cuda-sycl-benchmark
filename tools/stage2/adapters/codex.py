"""Codex CLI adapter with JSONL token telemetry and reproducible CLI discovery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def parse_codex_jsonl(text: str) -> dict[str, Any]:
    """Normalize the stable events emitted by ``codex exec --json``."""
    events = _objects(text)
    usages = [
        event["usage"]
        for event in events
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict)
    ]
    tokens_in = sum(_integer(item.get("input_tokens")) for item in usages)
    cached_input = sum(_integer(item.get("cached_input_tokens")) for item in usages)
    tokens_out = sum(_integer(item.get("output_tokens")) for item in usages)
    reasoning_output = sum(_integer(item.get("reasoning_output_tokens")) for item in usages)
    session = next(
        (event.get("thread_id") for event in events if event.get("type") == "thread.started"),
        None,
    )
    agent_messages = [
        item["text"]
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        for item in [event["item"]]
        if item.get("type") == "agent_message" and isinstance(item.get("text"), str)
    ]
    tool_types = {"command_execution", "mcp_tool_call", "file_change"}
    iterations = sum(
        event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") in tool_types
        for event in events
    )
    # Transient transport retries are emitted as ``error`` events even when the
    # turn later completes successfully. Only a terminal turn failure is fatal.
    failed = any(event.get("type") == "turn.failed" for event in events)
    completed = any(event.get("type") == "turn.completed" for event in events)
    return {
        "events": len(events),
        "status": "completed" if completed and not failed else "error",
        "session_id": str(session) if session else None,
        "iterations": int(iterations),
        "tokens_in": tokens_in if usages else None,
        "cached_input_tokens": min(cached_input, tokens_in) if usages else None,
        "uncached_input_tokens": max(tokens_in - cached_input, 0) if usages else None,
        "tokens_out": tokens_out if usages else None,
        "reasoning_output_tokens": reasoning_output if usages else None,
        "tokens_total": (tokens_in + tokens_out) if usages else None,
        "message": agent_messages[-1] if agent_messages else "",
    }


def _working_version(command: str) -> str | None:
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip() or "unknown"


def _windows_app_install_location() -> Path | None:
    if os.name != "nt":
        return None
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return None
    script = (
        "Get-AppxPackage -Name OpenAI.Codex | "
        "Sort-Object Version -Descending | Select-Object -First 1 "
        "-ExpandProperty InstallLocation"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return Path(value) if result.returncode == 0 and value else None


def resolve_codex_cli(command_name: str = "codex") -> tuple[str, str]:
    """Resolve an executable Codex CLI, including the Windows app package."""
    configured = os.environ.get("STAGE2_CODEX_COMMAND")
    candidates = [configured, command_name, shutil.which(command_name)]
    for candidate in candidates:
        if not candidate:
            continue
        version = _working_version(str(candidate))
        if version:
            return str(candidate), version

    install_location = _windows_app_install_location()
    packaged = install_location / "app" / "resources" / "codex.exe" if install_location else None
    if packaged and packaged.is_file():
        digest = hashlib.sha256(packaged.read_bytes()).hexdigest()[:16]
        cache = Path(tempfile.gettempdir()) / "stage2-codex-cli" / digest
        cache.mkdir(parents=True, exist_ok=True)
        copied = cache / "codex.exe"
        if not copied.is_file() or copied.stat().st_size != packaged.stat().st_size:
            shutil.copy2(packaged, copied)
        for helper_name in ("codex-windows-sandbox-setup.exe", "codex-command-runner.exe"):
            helper = packaged.parent / helper_name
            destination = cache / helper_name
            if helper.is_file() and (
                not destination.is_file() or destination.stat().st_size != helper.stat().st_size
            ):
                shutil.copy2(helper, destination)
        version = _working_version(str(copied))
        if version:
            return str(copied), version
    raise RuntimeError(
        "Codex CLI is unavailable. Install/authenticate Codex or set "
        "STAGE2_CODEX_COMMAND to an executable codex path."
    )


class CodexAdapter(HarnessAdapter):
    slug = "codex"

    def run(self, context: SessionContext) -> SessionResult:
        command, version = resolve_codex_cli(str(context.harness.get("command") or "codex"))
        reasoning_effort = str(context.harness.get("reasoning_effort") or "low")
        prompt, _ = materialize_agent_prompt(context)

        argv = [
            command,
            "exec",
            "--model",
            context.model_id,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if context.harness.get("sandbox_network_access", True):
            argv.extend(["-c", "sandbox_workspace_write.network_access=true"])
        if context.harness.get("externally_isolated_bypass"):
            argv.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            argv.extend(["--sandbox", "workspace-write"])
        argv.extend(
            [
                "--cd",
                str(context.sandbox_path),
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--disable",
                "remote_plugin",
                "--json",
                "--color",
                "never",
                prompt,
            ]
        )
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
        telemetry = parse_codex_jsonl(executed.stdout)
        status = telemetry["status"] if executed.status == "pass" else executed.status
        message = str(telemetry["message"] or executed.stderr[-1000:] or status)
        return SessionResult(
            status=status,
            tokens_in=telemetry["tokens_in"],
            tokens_out=telemetry["tokens_out"],
            tokens_total=telemetry["tokens_total"],
            wall_clock_s=elapsed,
            iterations=telemetry["iterations"],
            cost_usd=None,
            cached_input_tokens=telemetry["cached_input_tokens"],
            reasoning_output_tokens=telemetry["reasoning_output_tokens"],
            message=message,
            session_id=telemetry["session_id"],
            reported_model=None,
            raw_telemetry={
                "event_count": telemetry["events"],
                "uncached_input_tokens": telemetry["uncached_input_tokens"],
                "process_status": executed.status,
                "returncode": executed.returncode,
                "cli_version": version,
                "requested_reasoning_effort": reasoning_effort,
                "externally_isolated_bypass": bool(
                    context.harness.get("externally_isolated_bypass")
                ),
            },
        )


def create_adapter() -> CodexAdapter:
    return CodexAdapter()
