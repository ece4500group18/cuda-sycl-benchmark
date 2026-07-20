"""Canonical, auditable prompt shared by every Stage 2 harness adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from common import REPO_ROOT

if TYPE_CHECKING:
    from adapters.base import SessionContext


PROMPT_ROOT = REPO_ROOT / "benchmark" / "stage2" / "prompts"


def _prompt_part(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8").strip()


def build_agent_prompt(with_skill: bool) -> str:
    condition = "with_skill.txt" if with_skill else "oob.txt"
    return " ".join(
        (_prompt_part("base.txt"), _prompt_part(condition), _prompt_part("finish.txt"))
    )


def materialize_agent_prompt(context: SessionContext) -> tuple[str, Path]:
    """Write the exact model-visible prompt into the sandbox and audit artifacts."""
    prompt = build_agent_prompt(context.skill_path is not None)
    sandbox_prompt = context.sandbox_path / "AGENT_PROMPT.md"
    sandbox_prompt.write_text(prompt + "\n", encoding="utf-8", newline="\n")
    (context.run_path / "agent_prompt.txt").write_text(
        prompt + "\n", encoding="utf-8", newline="\n"
    )
    return prompt, sandbox_prompt
