#!/usr/bin/env python3
"""Build the original CUDA program for each case.

Uses metadata.build.cuda_build_command, executed from the case directory.
Output is logged to logs/cuda_compile.log and status.cuda_compile is updated.

If nvcc is not installed, every case is marked 'skipped_no_cuda_toolkit'.

Usage:
    python3 tools/build_cuda.py [--category easy] [--case vectorAdd]
"""

from __future__ import annotations

import argparse
import os
import sys

import _common as C


def build_one(case_dir, cuda_ready, skip_status, skip_message):
    meta = C.load_metadata(case_dir)
    cid = C.case_id_of(case_dir)
    log = C.log_path(case_dir, "cuda_compile.log")

    if not cuda_ready:
        C.set_status(meta, "cuda_compile", skip_status)
        C.save_metadata(case_dir, meta)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"{skip_message}\nCUDA compile skipped.\n")
        print(f"[skip] {cid}: {skip_status}")
        return

    # Ensure the conventional build output directory exists.
    os.makedirs(os.path.join(case_dir, "original", "build"), exist_ok=True)
    command = meta.get("build", {}).get("cuda_build_command", "")
    if not command:
        C.set_status(meta, "cuda_compile", "fail")
        C.save_metadata(case_dir, meta)
        print(f"[FAIL] {cid}: no cuda_build_command in metadata")
        return

    command = C.prepare_cuda_build_command(command)
    rc, _ = C.run_logged(command, case_dir, log, timeout=600)
    if rc == 0:
        C.set_status(meta, "cuda_compile", "pass")
        print(f"[ok]   {cid}: cuda compile")
    else:
        C.set_status(meta, "cuda_compile", "fail")
        print(f"[FAIL] {cid}: cuda compile rc={rc} (see {log})")
    C.save_metadata(case_dir, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category")
    ap.add_argument("--case")
    args = ap.parse_args()

    cuda_ready, skip_status, skip_message = C.cuda_toolchain_status()
    if not cuda_ready:
        print(f"NOTE: {skip_message} all cases -> {skip_status}")

    for case_dir in C.iter_cases(args.category, args.case):
        try:
            build_one(case_dir, cuda_ready, skip_status, skip_message)
        except Exception as exc:
            print(f"[ERROR] {C.case_id_of(case_dir)}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
