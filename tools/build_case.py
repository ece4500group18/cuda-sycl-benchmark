#!/usr/bin/env python3
"""Build original CUDA cases and write logs/build_result.json."""

from __future__ import annotations

import argparse
import sys

import _stage1_common as C


def build_one(case: C.CaseInfo, timeout: int) -> dict:
    command = C.build_command(case.metadata)
    if not command:
        result = C.command_result(case, "fail", command, message="missing CUDA build command")
        C.write_json(case.path / "logs" / "build_result.json", result)
        return result

    ready, skip_status, message = C.cuda_toolchain_status()
    if not ready:
        result = C.command_result(case, skip_status, command, message=message)
        C.write_json(case.path / "logs" / "build_result.json", result)
        return result

    (case.path / "original" / "build").mkdir(parents=True, exist_ok=True)
    prepared = C.prepare_cuda_build_command(command)
    rc, _, elapsed = C.run_logged(prepared, case.path, case.path / "logs" / "cuda_compile.log", timeout)
    status = "pass" if rc == 0 else "fail"
    result = C.command_result(case, status, prepared, elapsed, rc)
    C.write_json(case.path / "logs" / "build_result.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1

    failures = 0
    for case in cases:
        result = build_one(case, args.timeout)
        print(f"[{result['status']}] {case.case_id}: cuda build")
        if result["status"] == "fail":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
