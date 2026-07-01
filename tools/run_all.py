#!/usr/bin/env python3
"""Run Stage 1 CUDA validation or benchmarking workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys

import _stage1_common as C


def run_step(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(C.REPO_ROOT / "tools" / script), *extra]
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(C.REPO_ROOT)).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cuda_verify", "cuda_benchmark"], required=True)
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--allow-unverified", action="store_true")
    args = ap.parse_args()

    common = []
    if args.case:
        common += ["--case", args.case]

    if args.stage == "cuda_verify":
        rc = run_step("build_case.py", [*common, "--timeout", str(args.timeout)])
        if rc != 0:
            return rc
        rc = run_step("run_case.py", [*common, "--timeout", str(args.timeout)])
        if rc != 0:
            return rc
        return run_step("verify_case.py", [*common, "--timeout", str(args.timeout)])

    bench_args = [
        *common,
        "--timeout",
        str(args.timeout),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    if args.allow_unverified:
        bench_args.append("--allow-unverified")
    return run_step("benchmark_case.py", bench_args)


if __name__ == "__main__":
    sys.exit(main())
