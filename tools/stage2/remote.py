"""Resolve and validate SSH executor configuration without storing credentials."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

from common import safe_component


REMOTE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")
SSH_TARGET = re.compile(r"^[A-Za-z0-9_.@:-]+$")


def resolve_ssh_target(executor: dict[str, Any], require: bool = True) -> str:
    env_name = str(executor.get("target_env") or "")
    target = os.environ.get(env_name) if env_name else None
    target = target or executor.get("target")
    if not target:
        if require:
            suffix = f"; set {env_name}" if env_name else ""
            raise ValueError(f"SSH executor target is not configured{suffix}")
        return f"${{{env_name}}}" if env_name else "<unset>"
    target = str(target)
    if not SSH_TARGET.fullmatch(target):
        raise ValueError("SSH target contains unsupported characters; use an SSH config alias if needed")
    return target


def build_remote_config(
    executor: dict[str, Any],
    components: Iterable[str],
    extra_sycl_flags: list[str],
    require_target: bool = True,
) -> dict[str, Any]:
    target = resolve_ssh_target(executor, require=require_target)
    remote_root = str(executor.get("remote_root") or "/tmp/cuda-sycl-benchmark-stage2")
    if not REMOTE_PATH.fullmatch(remote_root) or ".." in remote_root.split("/"):
        raise ValueError("remote_root must be a safe absolute path without '..'")
    safe_parts = [safe_component(str(item), "remote workspace component") for item in components]
    remote_workspace = "/".join([remote_root.rstrip("/"), *safe_parts])
    return {
        "target": target,
        "remote_workspace": remote_workspace,
        "setup_command": str(
            executor.get("setup_command")
            or "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1"
        ),
        "device_selector": str(executor["device_selector"]),
        "compiler": str(executor.get("compiler") or "icpx"),
        "extra_sycl_flags": [str(item) for item in extra_sycl_flags],
        "timeout_s": float(executor.get("remote_timeout_s") or 3600),
        "ssh_command": str(executor.get("ssh_command") or "ssh"),
        "scp_command": str(executor.get("scp_command") or "scp"),
        "ssh_options": [
            str(item)
            for item in executor.get(
                "ssh_options",
                ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"],
            )
        ],
        "scp_options": [
            str(item)
            for item in executor.get(
                "scp_options",
                ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"],
            )
        ],
    }
