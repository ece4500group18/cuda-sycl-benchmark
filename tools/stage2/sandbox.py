"""Create a whitelist-only migration workspace for one session."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from common import REPO_ROOT, TASK_TEMPLATE, ensure_within


SYCL_BUILD_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
: "${CXX:=icpx}"
extra_flags=()
if [[ -n "${EXTRA_SYCL_FLAGS:-}" ]]; then
  read -r -a extra_flags <<< "$EXTRA_SYCL_FLAGS"
fi
"$CXX" -fsycl -O2 -std=c++17 main.sycl.cpp -o app "${extra_flags[@]}"
"""

SYCL_RUN_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
mkdir -p output
: "${ONEAPI_DEVICE_SELECTOR:=level_zero:gpu}"
export ONEAPI_DEVICE_SELECTOR
./app "${1:-output/sycl.txt}"
"""

REMOTE_SYCL_BUILD_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
: "${STAGE2_PYTHON:=python}"
"$STAGE2_PYTHON" remote_exec.py build
"""

REMOTE_SYCL_RUN_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
: "${STAGE2_PYTHON:=python}"
"$STAGE2_PYTHON" remote_exec.py run "${1:-output/sycl.txt}"
"""

CUDA_BUILD_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
nvcc -O2 -std=c++17 main.cu -o cuda_app
"""

CUDA_RUN_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail
mkdir -p output
./cuda_app "${1:-output/cuda.txt}"
"""


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | 0o111)


def restore_fixed_wrappers(
    destination: Path,
    include_cuda_tools: bool = False,
    remote_config: dict | None = None,
) -> None:
    """Restore evaluator-owned scripts after an agent session."""
    if remote_config is None:
        _write_executable(destination / "sycl_build.sh", SYCL_BUILD_WRAPPER)
        _write_executable(destination / "sycl_run.sh", SYCL_RUN_WRAPPER)
        (destination / "remote_exec.py").unlink(missing_ok=True)
        (destination / "remote_config.json").unlink(missing_ok=True)
    else:
        _write_executable(destination / "sycl_build.sh", REMOTE_SYCL_BUILD_WRAPPER)
        _write_executable(destination / "sycl_run.sh", REMOTE_SYCL_RUN_WRAPPER)
        shutil.copy2(Path(__file__).with_name("remote_exec.py"), destination / "remote_exec.py")
        (destination / "remote_config.json").write_text(
            json.dumps(remote_config, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    if include_cuda_tools:
        _write_executable(destination / "cuda_build.sh", CUDA_BUILD_WRAPPER)
        _write_executable(destination / "cuda_run.sh", CUDA_RUN_WRAPPER)


def create_sandbox(
    case_path: Path,
    destination: Path,
    include_cuda_tools: bool = False,
    skill_path: Path | None = None,
    remote_config: dict | None = None,
) -> Path:
    case_path = ensure_within(case_path, REPO_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    required = {
        case_path / "original" / "main.cu": destination / "main.cu",
        case_path / "original" / "CMakeLists.txt": destination / "CMakeLists.txt",
        TASK_TEMPLATE: destination / "TASK.md",
    }
    for source, target in required.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)

    (destination / "output").mkdir()
    restore_fixed_wrappers(destination, include_cuda_tools, remote_config)
    if skill_path is not None:
        skill_path = ensure_within(skill_path, REPO_ROOT)
        if not (skill_path / "SKILL.md").is_file():
            raise FileNotFoundError(skill_path / "SKILL.md")
        shutil.copytree(skill_path, destination / "skill")
    return destination
