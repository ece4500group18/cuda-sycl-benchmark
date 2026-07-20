"""Command execution abstraction used by adapters and the harness."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common import utc_now


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    status: str
    returncode: int | None
    elapsed_s: float
    stdout: str
    stderr: str
    timestamp: str

    def as_dict(self) -> dict:
        return asdict(self)


class LocalExecutor:
    kind = "local"

    def run(
        self,
        argv: Sequence[str | Path],
        cwd: Path,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        command = [str(item) for item in argv]
        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(key): str(value) for key, value in env.items()})
        started = time.perf_counter()
        timestamp = utc_now()
        try:
            process = subprocess.run(
                command,
                cwd=str(cwd),
                env=merged_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
            )
            status = "pass" if process.returncode == 0 else "fail"
            return CommandResult(
                command=command,
                cwd=str(cwd),
                status=status,
                returncode=process.returncode,
                elapsed_s=time.perf_counter() - started,
                stdout=process.stdout or "",
                stderr=process.stderr or "",
                timestamp=timestamp,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return CommandResult(
                command=command,
                cwd=str(cwd),
                status="timeout",
                returncode=None,
                elapsed_s=time.perf_counter() - started,
                stdout=stdout,
                stderr=stderr,
                timestamp=timestamp,
            )
