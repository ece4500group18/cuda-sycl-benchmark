"""Shared paths, serialization, and validation helpers for Stage 2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "stage2"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "stage2"
DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "stage2" / "datasets" / "cuda_verified_250.json"
DEFAULT_EXPERIMENT = REPO_ROOT / "benchmark" / "stage2" / "experiments" / "pilot_v1.json"
TASK_TEMPLATE = REPO_ROOT / "benchmark" / "stage2" / "TRANSLATION_TASK.md"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def resolve_bash() -> str | None:
    """Prefer Git Bash on Windows; System32 bash is WSL with a separate SSH home."""
    configured = os.environ.get("STAGE2_BASH")
    candidates: list[str | None] = [configured]
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.extend(
            [
                str(program_files / "Git" / "bin" / "bash.exe"),
                str(program_files / "Git" / "usr" / "bin" / "bash.exe"),
            ]
        )
    candidates.append(shutil.which("bash"))
    return next((item for item in candidates if item and Path(item).is_file()), None)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    """Atomically write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(value, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def safe_component(value: str, label: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"invalid {label} {value!r}; use letters, digits, dot, dash, or underscore")
    return value


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"path escapes expected root: {resolved} is not under {root_resolved}")
    return resolved


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot determine Git revision: {result.stderr.strip()}")
    return result.stdout.strip()


def hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
