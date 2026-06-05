#!/usr/bin/env python3
"""Build the migrated SYCL program for each case.

Prefers manual_sycl/ if it contains sources (a human-fixed version),
otherwise builds the SYCLomatic output in syclomatic/. Uses
metadata.build.sycl_build_command from the case directory and logs to
logs/sycl_compile.log, updating status.sycl_compile.

Skips with a precise reason when the SYCL compiler is missing or no migrated
sources exist.

Usage:
    python3 tools/build_sycl.py [--category easy] [--case vectorAdd]
"""

from __future__ import annotations

import argparse
import os
import sys

import _common as C

SYCL_EXTS = (".cpp", ".dp.cpp", ".cc", ".cxx")


def has_sources(d):
    return os.path.isdir(d) and any(
        f.endswith(SYCL_EXTS) for f in os.listdir(d)
    )


def build_one(case_dir, compiler):
    meta = C.load_metadata(case_dir)
    cid = C.case_id_of(case_dir)
    log = C.log_path(case_dir, "sycl_compile.log")

    if compiler is None:
        C.set_status(meta, "sycl_compile", "skipped_no_sycl_compiler")
        C.save_metadata(case_dir, meta)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("No SYCL compiler (icpx/icx/clang++) found; skipped.\n")
        print(f"[skip] {cid}: no SYCL compiler")
        return

    manual = os.path.join(case_dir, "manual_sycl")
    auto = os.path.join(case_dir, "syclomatic")
    if not has_sources(manual) and not has_sources(auto):
        C.set_status(meta, "sycl_compile", "skipped_not_migrated")
        C.save_metadata(case_dir, meta)
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("No migrated SYCL sources in manual_sycl/ or syclomatic/.\n")
        print(f"[skip] {cid}: not migrated")
        return

    os.makedirs(os.path.join(case_dir, "build_sycl"), exist_ok=True)
    command = meta.get("build", {}).get("sycl_build_command", "")
    if not command:
        C.set_status(meta, "sycl_compile", "fail")
        C.save_metadata(case_dir, meta)
        print(f"[FAIL] {cid}: no sycl_build_command in metadata")
        return

    rc, _ = C.run_logged(command, case_dir, log, timeout=600)
    if rc == 0:
        C.set_status(meta, "sycl_compile", "pass")
        print(f"[ok]   {cid}: sycl compile")
    else:
        C.set_status(meta, "sycl_compile", "fail")
        print(f"[FAIL] {cid}: sycl compile rc={rc} (see {log})")
    C.save_metadata(case_dir, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category")
    ap.add_argument("--case")
    args = ap.parse_args()

    compiler = C.find_sycl_compiler()
    if compiler is None:
        print("NOTE: no SYCL compiler found; all cases -> "
              "skipped_no_sycl_compiler")
    else:
        print(f"Using SYCL compiler: {compiler}")

    for case_dir in C.iter_cases(args.category, args.case):
        try:
            build_one(case_dir, compiler)
        except Exception as exc:
            print(f"[ERROR] {C.case_id_of(case_dir)}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
