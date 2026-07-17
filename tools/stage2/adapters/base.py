"""Stable protocol for agent harness adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from executor import LocalExecutor


@dataclass(frozen=True)
class SessionContext:
    case_id: str
    case_path: Path
    sandbox_path: Path
    run_path: Path
    prompt_path: Path
    harness: dict[str, Any]
    model: dict[str, Any]
    model_id: str
    skill_condition: dict[str, Any]
    skill_path: Path | None
    budget: dict[str, Any]
    executor: LocalExecutor


@dataclass(frozen=True)
class SessionResult:
    status: str
    tokens_in: int | None
    tokens_out: int | None
    tokens_total: int | None
    wall_clock_s: float
    iterations: int
    cost_usd: float | None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    cost_source: str | None = None
    pricing_snapshot: dict[str, Any] | None = None
    message: str = ""
    synthetic: bool = False
    session_id: str | None = None
    reported_model: str | None = None
    duration_api_s: float | None = None
    raw_telemetry: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HarnessAdapter(ABC):
    slug: str
    synthetic = False

    @abstractmethod
    def run(self, context: SessionContext) -> SessionResult:
        """Produce sandbox/main.sycl.cpp and normalized session telemetry."""


# Compatibility for third-party adapters written against the Stage 2 MVP.
MigrationAdapter = HarnessAdapter
