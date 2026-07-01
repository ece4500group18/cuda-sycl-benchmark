#!/usr/bin/env python3
"""Benchmark original CUDA cases and write logs/perf_result.json."""

from __future__ import annotations

import argparse
import sys
import time

import _stage1_common as C


def benchmark_one(
    case: C.CaseInfo,
    warmup: int,
    iterations: int,
    timeout: int,
    allow_unverified: bool,
) -> dict:
    verify_result_path = case.path / "logs" / "verify_result.json"
    if not allow_unverified and verify_result_path.is_file():
        verify_result = C.read_json(verify_result_path)
        if verify_result.get("status") != "pass":
            result = C.command_result(
                case,
                "skipped",
                C.run_command(case.metadata),
                message="verification result is not pass",
            )
            C.write_json(case.path / "logs" / "perf_result.json", result)
            return result
    elif not allow_unverified:
        result = C.command_result(
            case,
            "skipped",
            C.run_command(case.metadata),
            message="missing logs/verify_result.json; run tools/verify_case.py first",
        )
        C.write_json(case.path / "logs" / "perf_result.json", result)
        return result

    if not C.has_nvidia_gpu():
        result = C.command_result(case, "skipped", C.run_command(case.metadata), message="no usable NVIDIA GPU detected")
        C.write_json(case.path / "logs" / "perf_result.json", result)
        return result

    command = C.run_command(case.metadata)
    if not command:
        result = C.command_result(case, "fail", command, message="missing CUDA run command")
        C.write_json(case.path / "logs" / "perf_result.json", result)
        return result

    (case.path / "output").mkdir(parents=True, exist_ok=True)
    measured_ms: list[float] = []
    failures: list[dict] = []
    total = warmup + iterations
    for idx in range(total):
        is_warmup = idx < warmup
        run_no = idx + 1 if is_warmup else idx - warmup + 1
        phase = "warmup" if is_warmup else "measure"
        output_rel = f"output/cuda_perf_{phase}_{run_no}.txt"
        run_command = C.benchmark_output_command(command, output_rel)
        run_command = C.normalize_executable_command(run_command, case.path)
        log_path = case.path / "logs" / f"cuda_perf_{phase}_{run_no}.log"
        start = time.perf_counter()
        rc, _, _ = C.run_logged(run_command, case.path, log_path, timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if rc != 0:
            failures.append({"phase": phase, "run": run_no, "returncode": rc, "log": str(log_path)})
            break
        if not is_warmup:
            measured_ms.append(elapsed_ms)

    summary = C.runtime_summary_ms(measured_ms)
    hw = C.hardware_info()
    result = {
        "case_name": case.case_id,
        "status": "fail" if failures else "pass",
        "gpu_model": hw["gpu_model"],
        "cuda_version": hw["cuda_version"],
        "driver_version": hw["driver_version"],
        "compiler": hw["compiler"],
        "problem_size": case.metadata.get("input", {}).get("sizes", ""),
        "warmup_iterations": warmup,
        "measure_iterations": iterations,
        "runtime_ms_mean": summary["mean"],
        "runtime_ms_median": summary["median"],
        "runtime_ms_std": summary["std"],
        "metric_type": "runtime",
        "bandwidth_GBps": None,
        "throughput_items_per_sec": None,
        "gflops": None,
        "timestamp": C.utc_now(),
        "notes": "Process-level elapsed time; kernel-level timers can be added per case.",
        "runs_ms": measured_ms,
        "failures": failures,
    }
    C.write_json(case.path / "logs" / "perf_result.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case_id, folder name, or repository-relative case path")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--allow-unverified", action="store_true")
    args = ap.parse_args()

    cases = C.iter_cases(args.case)
    if args.case and not cases:
        print(f"No case matched {args.case!r}.", file=sys.stderr)
        return 1

    failures = 0
    for case in cases:
        result = benchmark_one(case, args.warmup, args.iterations, args.timeout, args.allow_unverified)
        print(f"[{result['status']}] {case.case_id}: cuda benchmark")
        if result["status"] == "fail":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
