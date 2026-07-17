"""Offline test double for exercising the harness without SYCL hardware."""

from __future__ import annotations

import time

from adapters.base import HarnessAdapter, SessionContext, SessionResult


MOCK_SOURCE = """// Synthetic Stage 2 harness fixture. This is not a migration result.
#include <sycl/sycl.hpp>
int main() { return 0; }
"""


class MockAdapter(HarnessAdapter):
    slug = "mock"
    synthetic = True

    def run(self, context: SessionContext) -> SessionResult:
        started = time.perf_counter()
        (context.sandbox_path / "main.sycl.cpp").write_text(
            MOCK_SOURCE, encoding="utf-8", newline="\n"
        )
        return SessionResult(
            status="completed",
            tokens_in=0,
            tokens_out=0,
            tokens_total=0,
            wall_clock_s=time.perf_counter() - started,
            iterations=1,
            cost_usd=0.0,
            message="offline test double; no CUDA-to-SYCL migration was attempted",
            synthetic=True,
        )


def create_adapter() -> MockAdapter:
    return MockAdapter()
