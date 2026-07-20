"""Probe oneAPI, Level Zero, Linux device access, and a real SYCL kernel."""

from __future__ import annotations

import getpass
import json
import os
import platform
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import utc_now
from executor import LocalExecutor
from remote import build_remote_config, resolve_ssh_target


HELLO_SYCL = r"""#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
  try {
    sycl::queue q{sycl::gpu_selector_v, [](sycl::exception_list errors) {
      for (const auto &error : errors) {
        try { std::rethrow_exception(error); }
        catch (const sycl::exception &e) { std::cerr << e.what() << "\n"; }
      }
    }};
    auto d = q.get_device();
    int *value = sycl::malloc_shared<int>(1, q);
    if (!value) return 3;
    *value = 0;
    q.single_task([=] { *value = 450; }).wait_and_throw();
    std::cout << "vendor=" << d.get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "device=" << d.get_info<sycl::info::device::name>() << "\n";
    std::cout << "backend=" << static_cast<int>(q.get_backend()) << "\n";
    std::cout << "value=" << *value << "\n";
    if (argc > 1) {
      std::ofstream out(argv[1]);
      out << *value << "\n";
    }
    sycl::free(value, q);
    return 0;
  } catch (const sycl::exception &e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
}
"""


def _tool(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {"found": path is not None, "path": path}


def _version(executor: LocalExecutor, tool: dict[str, Any], args: list[str]) -> dict[str, Any]:
    if not tool["found"]:
        return {"status": "skipped", "reason": "tool not found"}
    return executor.run([str(tool["path"]), *args], Path.cwd(), 30).as_dict()


def probe_environment(run_smoke: bool = True, device_selector: str = "level_zero:gpu") -> dict[str, Any]:
    executor = LocalExecutor()
    names = ("icpx", "sycl-ls", "zeinfo", "clinfo", "lspci", "git", "python3")
    tools = {name: _tool(name) for name in names}
    result: dict[str, Any] = {
        "schema_version": 2,
        "timestamp": utc_now(),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "hostname": platform.node(),
            "user": getpass.getuser(),
        },
        "tools": tools,
        "tool_versions": {
            "icpx": _version(executor, tools["icpx"], ["--version"]),
            "sycl-ls": _version(executor, tools["sycl-ls"], ["--version"]),
        },
        "device_selector": device_selector,
        "render_nodes": [],
        "kernel_drivers": {"xe": Path("/sys/module/xe").exists(), "i915": Path("/sys/module/i915").exists()},
        "sycl_ls": {"status": "skipped", "reason": "sycl-ls not found"},
        "zeinfo": {"status": "skipped", "reason": "zeinfo not found (optional diagnostic)"},
        "clinfo": {"status": "skipped", "reason": "clinfo not found (optional diagnostic)"},
        "lspci": {"status": "skipped", "reason": "lspci not found (optional diagnostic)"},
        "compile_smoke": {"status": "skipped", "reason": "icpx not found"},
        "run_smoke": {"status": "skipped", "reason": "compile smoke not run"},
        "ready_for_intel_gpu_experiments": False,
        "blocking_reasons": [],
    }

    if os.name != "nt":
        render_nodes = []
        for path in sorted(Path("/dev/dri").glob("renderD*")):
            stat = path.stat()
            try:
                import grp

                group = grp.getgrgid(stat.st_gid).gr_name
            except (ImportError, KeyError):
                group = str(stat.st_gid)
            render_nodes.append(
                {
                    "path": str(path), "group": group,
                    "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK),
                }
            )
        result["render_nodes"] = render_nodes

    if tools["sycl-ls"]["found"]:
        ls_result = executor.run(
            [str(tools["sycl-ls"]["path"]), "--ignore-device-selectors"], Path.cwd(), 30
        )
        if ls_result.status != "pass" and "unknown" in (ls_result.stderr + ls_result.stdout).lower():
            ls_result = executor.run([str(tools["sycl-ls"]["path"])], Path.cwd(), 30)
        result["sycl_ls"] = ls_result.as_dict()
    for name in ("zeinfo", "clinfo"):
        if tools[name]["found"]:
            result[name] = executor.run([str(tools[name]["path"])], Path.cwd(), 30).as_dict()
    if tools["lspci"]["found"]:
        result["lspci"] = executor.run([str(tools["lspci"]["path"]), "-nnk"], Path.cwd(), 30).as_dict()

    if run_smoke and tools["icpx"]["found"]:
        with tempfile.TemporaryDirectory(prefix="stage2-doctor-") as tmp:
            workdir = Path(tmp)
            source = workdir / "hello_sycl.cpp"
            binary = workdir / ("hello_sycl.exe" if os.name == "nt" else "hello_sycl")
            source.write_text(HELLO_SYCL, encoding="utf-8", newline="\n")
            compiled = executor.run(
                [str(tools["icpx"]["path"]), "-fsycl", "-std=c++17", str(source), "-o", str(binary)],
                workdir, 120,
            )
            result["compile_smoke"] = compiled.as_dict()
            if compiled.status == "pass":
                result["run_smoke"] = executor.run(
                    [str(binary)], workdir, 120, env={"ONEAPI_DEVICE_SELECTOR": device_selector}
                ).as_dict()
    elif not run_smoke:
        result["compile_smoke"] = {"status": "skipped", "reason": "disabled by --no-smoke"}
        result["run_smoke"] = {"status": "skipped", "reason": "disabled by --no-smoke"}

    sycl_text = str(result["sycl_ls"].get("stdout", "")) + str(result["sycl_ls"].get("stderr", ""))
    run_text = str(result["run_smoke"].get("stdout", "")) + str(result["run_smoke"].get("stderr", ""))
    checks = {
        "icpx_compile": result["compile_smoke"].get("status") == "pass",
        "intel_gpu_visible_to_sycl": "intel" in sycl_text.lower() and "gpu" in sycl_text.lower(),
        "level_zero_visible": "level_zero" in sycl_text.lower(),
        "intel_gpu_kernel_runs": result["run_smoke"].get("status") == "pass"
        and "intel" in run_text.lower() and "value=450" in run_text,
    }
    result["readiness_checks"] = checks
    result["blocking_reasons"] = [name for name, passed in checks.items() if not passed]
    result["ready_for_intel_gpu_experiments"] = all(checks.values())
    return result


def _remote_shell(
    executor: LocalExecutor,
    config: dict[str, Any],
    script: str,
    timeout_s: float = 120,
) -> dict[str, Any]:
    command = [
        str(config["ssh_command"]),
        *[str(item) for item in config["ssh_options"]],
        str(config["target"]),
        f"bash -lc {shlex.quote(script)}",
    ]
    return executor.run(command, Path.cwd(), timeout_s).as_dict()


def probe_ssh_environment(executor_config: dict[str, Any], run_smoke: bool = True) -> dict[str, Any]:
    """Probe a remote Intel host while keeping the repository and verifier local."""
    executor = LocalExecutor()
    target = resolve_ssh_target(executor_config)
    config = build_remote_config(
        executor_config,
        ("doctor", f"process-{os.getpid()}"),
        [],
    )
    setup = str(config["setup_command"])
    result: dict[str, Any] = {
        "schema_version": 2,
        "timestamp": utc_now(),
        "executor_kind": "ssh",
        "target": target,
        "device_selector": config["device_selector"],
        "local_tools": {
            "ssh": _tool(str(config["ssh_command"])),
            "scp": _tool(str(config["scp_command"])),
        },
        "remote_identity": _remote_shell(executor, config, "hostname; id"),
        "render_nodes": _remote_shell(executor, config, "ls -l /dev/dri/renderD*"),
        "tool_versions": _remote_shell(
            executor,
            config,
            f"set -e; {setup}; icpx --version; command -v sycl-ls",
        ),
        "sycl_ls": _remote_shell(
            executor,
            config,
            f"set -e; {setup}; sycl-ls --ignore-device-selectors",
        ),
        "compile_smoke": {"status": "skipped", "reason": "disabled by --no-smoke"},
        "run_smoke": {"status": "skipped", "reason": "disabled by --no-smoke"},
        "ready_for_intel_gpu_experiments": False,
        "blocking_reasons": [],
    }
    if run_smoke:
        with tempfile.TemporaryDirectory(prefix="stage2-remote-doctor-") as tmp:
            workdir = Path(tmp)
            (workdir / "main.sycl.cpp").write_text(
                HELLO_SYCL, encoding="utf-8", newline="\n"
            )
            shutil.copy2(Path(__file__).with_name("remote_exec.py"), workdir / "remote_exec.py")
            (workdir / "remote_config.json").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            result["compile_smoke"] = executor.run(
                [sys.executable, str(workdir / "remote_exec.py"), "build"],
                workdir,
                300,
            ).as_dict()
            if result["compile_smoke"]["status"] == "pass":
                result["run_smoke"] = executor.run(
                    [
                        sys.executable,
                        str(workdir / "remote_exec.py"),
                        "run",
                        str(workdir / "probe.txt"),
                    ],
                    workdir,
                    300,
                ).as_dict()
    sycl_text = str(result["sycl_ls"].get("stdout", "")) + str(
        result["sycl_ls"].get("stderr", "")
    )
    run_text = str(result["run_smoke"].get("stdout", "")) + str(
        result["run_smoke"].get("stderr", "")
    )
    checks = {
        "ssh_authentication": result["remote_identity"].get("status") == "pass",
        "intel_gpu_visible_to_sycl": "intel" in sycl_text.lower()
        and "gpu" in sycl_text.lower(),
        "level_zero_visible": "level_zero" in sycl_text.lower(),
        "remote_compile": result["compile_smoke"].get("status") == "pass",
        "remote_intel_kernel_runs": result["run_smoke"].get("status") == "pass"
        and "intel" in run_text.lower()
        and "value=450" in run_text,
    }
    result["readiness_checks"] = checks
    result["blocking_reasons"] = [name for name, passed in checks.items() if not passed]
    result["ready_for_intel_gpu_experiments"] = all(checks.values())
    workspace = str(config["remote_workspace"])
    result["cleanup"] = _remote_shell(
        executor,
        config,
        f"rm -rf -- {shlex.quote(workspace)}",
    )
    return result
