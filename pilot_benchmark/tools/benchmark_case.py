#!/usr/bin/env python3
"""Collect end-to-end runtime measurements for built and verified cases.

This is a process-level performance smoke benchmark: it times the benchmark
binary from launch to exit, including deterministic input generation and output
file writing. That is not a pure kernel-time metric, but it is uniform across
original CUDA and migrated SYCL variants and gives the migration evaluator a
repeatable baseline until individual cases expose device-side timing.

Usage:
    python3 tools/benchmark_case.py --variant cuda [--repeat 5 --warmup 1]
    python3 tools/benchmark_case.py --variant sycl --category easy
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import _common as C


def command_with_output(command, variant, output_rel):
    """Use a benchmark-specific output file if the command has an output arg."""
    default_posix = f"output/{variant}_output.txt"
    default_win = f"output\\{variant}_output.txt"
    if default_posix in command:
        return command.replace(default_posix, output_rel)
    if default_win in command:
        return command.replace(default_win, output_rel.replace("/", "\\"))
    return f"{command} {output_rel}"


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def summarize(times):
    if not times:
        return {}
    return {
        "unit": "seconds",
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def skip(case_dir, meta, variant, status, reason):
    cid = C.case_id_of(case_dir)
    key = f"{variant}_performance"
    C.set_status(meta, key, status)
    C.save_metadata(case_dir, meta)
    out_path = os.path.join(case_dir, "output", f"{variant}_performance.json")
    write_json(out_path, {
        "case_id": cid,
        "category": C.category_of(case_dir),
        "variant": variant,
        "status": status,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[skip] {cid}: {variant} performance ({status})")


def benchmark_one(case_dir, variant, repeat, warmup, timeout, allow_unverified):
    meta = C.load_metadata(case_dir)
    cid = C.case_id_of(case_dir)
    status = meta.get("status", {})
    perf_key = f"{variant}_performance"
    verify_key = f"{variant}_verify"
    compile_key = f"{variant}_compile"

    if status.get(compile_key) != "pass":
        skip(case_dir, meta, variant, "skipped_not_built",
             f"{compile_key} is {status.get(compile_key, 'unknown')}")
        return

    if not allow_unverified and status.get(verify_key) != "pass":
        skip(case_dir, meta, variant, "skipped_not_verified",
             f"{verify_key} is {status.get(verify_key, 'unknown')}")
        return

    if variant == "cuda" and not C.has_nvidia_gpu():
        skip(case_dir, meta, variant, "skipped_no_cuda_gpu", "no NVIDIA GPU")
        return
    if variant == "sycl" and not C.has_sycl_device():
        skip(case_dir, meta, variant, "skipped_no_sycl_device", "no SYCL device")
        return

    command = meta.get("run", {}).get(f"{variant}_run_command", "")
    if not command:
        C.set_status(meta, perf_key, "fail")
        C.save_metadata(case_dir, meta)
        print(f"[FAIL] {cid}: no {variant}_run_command")
        return

    os.makedirs(os.path.join(case_dir, "output"), exist_ok=True)
    times = []
    failures = []

    total_runs = warmup + repeat
    for idx in range(total_runs):
        phase = "warmup" if idx < warmup else "run"
        run_no = idx + 1 if phase == "warmup" else idx - warmup + 1
        output_rel = f"output/{variant}_perf_{phase}_{run_no}.txt"
        cmd = command_with_output(command, variant, output_rel)
        cmd = C.normalize_executable_command(cmd, case_dir)
        log = C.log_path(case_dir, f"{variant}_perf_{phase}_{run_no}.log")

        start = time.perf_counter()
        rc, _ = C.run_logged(cmd, case_dir, log, timeout=timeout)
        elapsed = time.perf_counter() - start
        if rc != 0:
            failures.append({"phase": phase, "run": run_no, "rc": rc, "log": log})
            break
        if phase == "run":
            times.append(elapsed)

    result = {
        "case_id": cid,
        "category": C.category_of(case_dir),
        "variant": variant,
        "metric": "end_to_end_process_runtime",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repeat": repeat,
        "warmup": warmup,
        "command": command,
        "runs_seconds": times,
        "summary": summarize(times),
        "failures": failures,
        "status": "fail" if failures else "pass",
    }
    write_json(os.path.join(case_dir, "output", f"{variant}_performance.json"), result)

    C.set_status(meta, perf_key, result["status"])
    C.save_metadata(case_dir, meta)
    if failures:
        print(f"[FAIL] {cid}: {variant} performance")
    else:
        med = result["summary"].get("median", 0.0)
        print(f"[ok]   {cid}: {variant} performance median={med:.6f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cuda", "sycl"], required=True)
    ap.add_argument("--category")
    ap.add_argument("--case")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--allow-unverified", action="store_true")
    args = ap.parse_args()

    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be >= 0", file=sys.stderr)
        return 2

    for case_dir in C.iter_cases(args.category, args.case):
        try:
            benchmark_one(
                case_dir, args.variant, args.repeat, args.warmup,
                args.timeout, args.allow_unverified,
            )
        except Exception as exc:
            print(f"[ERROR] {C.case_id_of(case_dir)}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
