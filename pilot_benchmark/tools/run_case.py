#!/usr/bin/env python3
"""Run a built case binary (CUDA or SYCL variant) and record run status.

The metadata run command is expected to write the program's numerical result
to output/<variant>_output.txt so that verify_case.py can check it later.

Skips (without failing) when:
  - the prerequisite build did not pass            -> skipped_not_built
  - no NVIDIA GPU is present (cuda variant)         -> skipped_no_cuda_gpu
  - no SYCL device is present (sycl variant)        -> skipped_no_sycl_device

Usage:
    python3 tools/run_case.py --variant cuda [--category easy] [--case vectorAdd]
    python3 tools/run_case.py --variant sycl
"""

from __future__ import annotations

import argparse
import os
import sys

import _common as C


def run_one(case_dir, variant, device_ok):
    meta = C.load_metadata(case_dir)
    cid = C.case_id_of(case_dir)
    status = meta.get("status", {})

    if variant == "cuda":
        run_key, compile_key = "cuda_run", "cuda_compile"
        log = C.log_path(case_dir, "cuda_run.log")
        command = meta.get("run", {}).get("cuda_run_command", "")
        no_device = "skipped_no_cuda_gpu"
    else:
        run_key, compile_key = "sycl_run", "sycl_compile"
        log = C.log_path(case_dir, "sycl_run.log")
        command = meta.get("run", {}).get("sycl_run_command", "")
        no_device = "skipped_no_sycl_device"

    if status.get(compile_key) != "pass":
        C.set_status(meta, run_key, "skipped_not_built")
        C.save_metadata(case_dir, meta)
        print(f"[skip] {cid}: {variant} not built ({status.get(compile_key)})")
        return

    if not device_ok:
        C.set_status(meta, run_key, no_device)
        C.save_metadata(case_dir, meta)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"No usable {variant} device; run skipped.\n")
        print(f"[skip] {cid}: {no_device}")
        return

    if not command:
        C.set_status(meta, run_key, "fail")
        C.save_metadata(case_dir, meta)
        print(f"[FAIL] {cid}: no {variant}_run_command")
        return

    # The run command writes its result to output/<variant>_output.txt; make
    # sure that directory exists (it is gitignored and absent on a fresh clone).
    os.makedirs(os.path.join(case_dir, "output"), exist_ok=True)

    command = C.normalize_executable_command(command, case_dir)
    rc, _ = C.run_logged(command, case_dir, log, timeout=600)
    C.set_status(meta, run_key, "pass" if rc == 0 else "fail")
    C.save_metadata(case_dir, meta)
    print(f"[{'ok' if rc == 0 else 'FAIL'}]   {cid}: {variant} run rc={rc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["cuda", "sycl"], required=True)
    ap.add_argument("--category")
    ap.add_argument("--case")
    args = ap.parse_args()

    if args.variant == "cuda":
        device_ok = C.has_nvidia_gpu()
        if not device_ok:
            print("NOTE: no NVIDIA GPU -> cuda runs skipped_no_cuda_gpu")
    else:
        device_ok = C.has_sycl_device()
        if not device_ok:
            print("NOTE: no SYCL device -> sycl runs skipped_no_sycl_device")

    for case_dir in C.iter_cases(args.category, args.case):
        try:
            run_one(case_dir, args.variant, device_ok)
        except Exception as exc:
            print(f"[ERROR] {C.case_id_of(case_dir)}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
