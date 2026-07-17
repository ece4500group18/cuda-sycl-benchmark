#!/usr/bin/env python3
"""Standalone SSH build/run proxy copied into each migration sandbox."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "remote_config.json"


def _load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("remote_config.json must contain an object")
    for key in ("target", "remote_workspace", "setup_command", "device_selector"):
        if not value.get(key):
            raise ValueError(f"remote_config.json is missing {key}")
    return value


def _run(argv: list[str], timeout_s: float) -> int:
    try:
        completed = subprocess.run(argv, cwd=ROOT, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"remote command timed out after {timeout_s:.0f}s", file=sys.stderr)
        return 124
    return int(completed.returncode)


def _ssh_argv(config: dict[str, Any], script: str) -> list[str]:
    return [
        str(config.get("ssh_command") or "ssh"),
        *[str(item) for item in config.get("ssh_options", [])],
        str(config["target"]),
        f"bash -lc {shlex.quote(script)}",
    ]


def _scp_argv(config: dict[str, Any], source: str, destination: str) -> list[str]:
    return [
        str(config.get("scp_command") or "scp"),
        *[str(item) for item in config.get("scp_options", [])],
        source,
        destination,
    ]


def _preamble(config: dict[str, Any]) -> str:
    workspace = shlex.quote(str(config["remote_workspace"]))
    setup = str(config["setup_command"])
    # Intel's setvars.sh is not guaranteed to be nounset-safe. Enable strict
    # error/pipe handling only after the vendor environment has been sourced.
    return f"set -e; {setup}; set -eo pipefail; mkdir -p {workspace}; cd {workspace}"


def build(config: dict[str, Any]) -> int:
    source = ROOT / "main.sycl.cpp"
    if not source.is_file() or source.is_symlink():
        print("main.sycl.cpp is missing or is a symlink", file=sys.stderr)
        return 2
    timeout_s = float(config.get("timeout_s") or 3600)
    workspace = str(config["remote_workspace"])
    mkdir_script = f"{_preamble(config)}; true"
    status = _run(_ssh_argv(config, mkdir_script), timeout_s)
    if status:
        return status
    status = _run(
        _scp_argv(config, str(source), f"{config['target']}:{workspace}/main.sycl.cpp"),
        timeout_s,
    )
    if status:
        return status
    flags = ["-fsycl", "-O2", "-std=c++17", "main.sycl.cpp", "-o", "app"]
    flags.extend(str(item) for item in config.get("extra_sycl_flags", []))
    compiler = shlex.quote(str(config.get("compiler") or "icpx"))
    compile_command = " ".join([compiler, *[shlex.quote(item) for item in flags]])
    return _run(_ssh_argv(config, f"{_preamble(config)}; {compile_command}"), timeout_s)


def run(config: dict[str, Any], local_output: Path) -> int:
    timeout_s = float(config.get("timeout_s") or 3600)
    selector = shlex.quote(str(config["device_selector"]))
    remote_output = "output/sycl.txt"
    script = (
        f"{_preamble(config)}; mkdir -p output; "
        f"ONEAPI_DEVICE_SELECTOR={selector} ./app {shlex.quote(remote_output)}"
    )
    status = _run(_ssh_argv(config, script), timeout_s)
    if status:
        return status
    local_output.parent.mkdir(parents=True, exist_ok=True)
    workspace = str(config["remote_workspace"])
    return _run(
        _scp_argv(
            config,
            f"{config['target']}:{workspace}/{remote_output}",
            str(local_output),
        ),
        timeout_s,
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"build", "run"}:
        print("usage: remote_exec.py build | run [local-output]", file=sys.stderr)
        return 2
    config = _load_config()
    if sys.argv[1] == "build":
        return build(config)
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "output" / "sycl.txt"
    return run(config, output.resolve())


if __name__ == "__main__":
    sys.exit(main())
