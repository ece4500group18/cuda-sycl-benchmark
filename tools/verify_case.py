#!/usr/bin/env python3
"""Verify original CUDA output with the case's tests/verify.py."""

from __future__ import annotations

import argparse
import sys

import _stage1_common as C


def verify_one(case: C.CaseInfo, timeout: int) -> dict:
    output_rel = "output/cuda_output.txt"
    output_path = case.path / output_rel
    command = C.python_command("tests/verify.py", "--variant", "cuda", "--output", output_rel)
    for filename in ("build_result.json", "run_result.json"):
        status_path = case.path / "logs" / filename
        if status_path.is_file():
            prior = C.read_json(status_path)
            if prior.get("status") != "pass":
                result = C.command_result(case, "skipped", command, message=f"{filename} is not pass")
                C.write_json(case.path / "logs" / "verify_result.json", result)
                return result
    if not output_path.is_file():
        result = C.command_result(case, "skipped", command, message=f"missing {output_rel}")
        C.write_json(case.path / "logs" / "verify_result.json", result)
        return result
    rc, output, elapsed = C.run_logged(command, case.path, case.path / "logs" / "verify.log", timeout)
    passed = rc == 0 and "PASS" in output
    result = C.command_result(case, "pass" if passed else "fail", command, elapsed, rc)
    result["verifier_output"] = output[-4000:]
    C.write_json(case.path / "logs" / "verify_result.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1

    failures = 0
    for case in cases:
        result = verify_one(case, args.timeout)
        print(f"[{result['status']}] {case.case_id}: cuda verify")
        if result["status"] == "fail":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
