"""Portable adapter for harness CLIs that are configured with an argv template."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from adapters.base import HarnessAdapter, SessionContext, SessionResult
from common import ensure_within, resolve_bash
from prompts import materialize_agent_prompt


class ExternalCommandAdapter(HarnessAdapter):
    slug = "external-command"

    def run(self, context: SessionContext) -> SessionResult:
        template = context.harness.get("argv")
        if not isinstance(template, list) or not template:
            raise ValueError("external_command harness requires a non-empty argv array")
        prompt, agent_prompt_path = materialize_agent_prompt(context)
        replacements = {
            "sandbox": str(context.sandbox_path),
            "prompt_file": str(context.prompt_path),
            "agent_prompt_file": str(agent_prompt_path),
            "prompt": prompt,
            "model_id": context.model_id,
            "skill_file": str(context.skill_path or ""),
        }
        argv = []
        for item in template:
            rendered = str(item)
            for key, value in replacements.items():
                rendered = rendered.replace("{" + key + "}", value)
            argv.append(rendered)
        executable = shutil.which(argv[0])
        if not executable:
            raise RuntimeError(f"harness executable not found: {argv[0]}")
        argv[0] = executable
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
        (context.run_path / "harness_stdout.log").write_text(
            executed.stdout, encoding="utf-8", newline="\n"
        )
        (context.run_path / "harness_stderr.log").write_text(
            executed.stderr, encoding="utf-8", newline="\n"
        )
        telemetry: dict[str, Any] = {}
        telemetry_name = context.harness.get("telemetry_file")
        if telemetry_name:
            telemetry_path = ensure_within(context.sandbox_path / str(telemetry_name), context.sandbox_path)
            if telemetry_path.is_file():
                value = json.loads(telemetry_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    telemetry = value
        tokens_in = telemetry.get("tokens_in")
        tokens_out = telemetry.get("tokens_out")
        total = telemetry.get("tokens_total")
        if total is None and isinstance(tokens_in, int) and isinstance(tokens_out, int):
            total = tokens_in + tokens_out
        return SessionResult(
            status="completed" if executed.status == "pass" else executed.status,
            tokens_in=tokens_in if isinstance(tokens_in, int) else None,
            tokens_out=tokens_out if isinstance(tokens_out, int) else None,
            tokens_total=total if isinstance(total, int) else None,
            wall_clock_s=elapsed,
            iterations=int(telemetry.get("iterations") or 0),
            cost_usd=float(telemetry["cost_usd"]) if telemetry.get("cost_usd") is not None else None,
            message=str(telemetry.get("message") or executed.stderr[-1000:]),
            session_id=telemetry.get("session_id"),
            reported_model=telemetry.get("model"),
            raw_telemetry={"returncode": executed.returncode, "process_status": executed.status},
        )


def create_adapter() -> ExternalCommandAdapter:
    return ExternalCommandAdapter()
