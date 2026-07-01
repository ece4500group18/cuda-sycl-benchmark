#!/usr/bin/env python3
"""Run original CUDA cases and write logs/run_result.json."""

from __future__ import annotations

import argparse
import sys

import _stage1_common as C


def run_one(case: C.CaseInfo, timeout: int, allow_no_gpu: bool = False) -> dict:
    command = C.run_command(case.metadata)
    if not command:
        result = C.command_result(case, "fail", command, message="missing CUDA run command")
        C.write_json(case.path / "logs" / "run_result.json", result)
        return result
    build_log = case.path / "logs" / "build_result.json"
    if build_log.is_file():
        build_result = C.read_json(build_log)
        if build_result.get("status") != "pass":
            result = C.command_result(case, "skipped", command, message="build_result.json is not pass")
            C.write_json(case.path / "logs" / "run_result.json", result)
            return result
    if not allow_no_gpu and not C.has_nvidia_gpu():
        result = C.command_result(case, "skipped", command, message="no usable NVIDIA GPU detected")
        C.write_json(case.path / "logs" / "run_result.json", result)
        return result

    (case.path / "output").mkdir(parents=True, exist_ok=True)
    prepared = C.normalize_executable_command(command, case.path)
    rc, _, elapsed = C.run_logged(prepared, case.path, case.path / "logs" / "cuda_run.log", timeout)
    status = "pass" if rc == 0 else "fail"
    result = C.command_result(case, status, prepared, elapsed, rc)
    C.write_json(case.path / "logs" / "run_result.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--allow-no-gpu", action="store_true", help="attempt the run even if nvidia-smi is absent")
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1

    failures = 0
    for case in cases:
        result = run_one(case, args.timeout, args.allow_no_gpu)
        print(f"[{result['status']}] {case.case_id}: cuda run")
        if result["status"] == "fail":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
