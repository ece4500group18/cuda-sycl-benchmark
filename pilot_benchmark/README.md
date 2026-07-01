# CUDA-to-SYCL Migration Pilot Benchmark

A small, **runnable and verifiable** pilot dataset for evaluating automated
CUDA→SYCL migration (SYCLomatic baseline + later agent-based migration).

The goal is **not** size. The goal is that, for every case, the framework can
answer concrete yes/no questions:

- Can the original CUDA case be automatically compiled / run / verified?
- Is the metadata complete enough for later agent-based migration?
- Can the SYCLomatic baseline migration be executed?
- Can the generated SYCL code be compiled / run / verified against a reference?

Each case is a **complete benchmark unit** (source + build command + run
command + correctness rule + metadata + logs), in the spirit of NVIDIA
cuda-samples — not a bare code snippet.

## Status

All 50 cases are implemented — 10 easy, 10 medium (shared mem / reduction),
10 HPC, 10 AI kernels, 10 library/API — and the full
build → run → verify → migrate → build → run → verify pipeline runs
end-to-end. See `reports/pilot_status.md`.

Every case's `tests/verify.py` is independently validated with
`--selftest` (it writes its own CPU reference, then verifies it), so the
correctness layer works even on machines without a CUDA/SYCL toolchain.
On such a machine all build/run/migrate stages report `skipped_*`; install
the toolchains (see below) and re-run the scripts to get real pass/fail.

## Repository layout

```
pilot_benchmark/
  README.md
  cases/<category>/<case_id>/      easy | medium | hpc | ai | library_api
    original/      main.cu, CMakeLists.txt, README.md
    syclomatic/    SYCLomatic-generated SYCL output
    manual_sycl/   optional hand-fixed SYCL version (preferred by build_sycl)
    input/         optional input data / generators
    output/        runtime + reference outputs (cuda_output.txt, sycl_output.txt)
    logs/          *_compile.log, *_run.log, syclomatic.log, verify.log
    tests/verify.py
    metadata.json
    README.md
  tools/           collect_cases, run_syclomatic, build_cuda, build_sycl,
                   run_case, verify_case, generate_report (+ _common, verify_lib)
  reports/         pilot_status.csv, pilot_status.md
  scripts/         setup_env.sh, run_all_cuda.sh, run_all_syclomatic.sh,
                   run_all_sycl.sh, verify_all.sh
```

`tools/metadata_schema.json` is the JSON Schema every `metadata.json` follows.

## Dependencies

| purpose | tool | required? |
| --- | --- | --- |
| CUDA build | `nvcc` (CUDA Toolkit) | for cuda_compile/run |
| CUDA run | NVIDIA GPU + driver | for cuda_run/verify |
| migration | `c2s` (SYCLomatic) or `dpct` (oneAPI) | for syclomatic_migrate |
| SYCL build | `icpx` / `icx` / `clang++` (DPC++) | for sycl_compile |
| SYCL run | a SYCL device (`sycl-ls`) | for sycl_run/verify |
| verification | `python3` + `numpy` | always |

**Graceful degradation:** missing toolchains/devices never abort the
pipeline. The affected step is recorded as a `skipped_*` status (see the
vocabulary at the top of `tools/_common.py`) and shown in the report.

Check what is available:

```bash
bash scripts/setup_env.sh
```

## How to run

```bash
# 0. validate case structure + metadata
python3 tools/collect_cases.py

# 1. baseline migration (CUDA -> SYCL via SYCLomatic)
bash scripts/run_all_syclomatic.sh

# 2. original CUDA: build, run, verify
bash scripts/run_all_cuda.sh

# 3. migrated SYCL: build, run, verify
bash scripts/run_all_sycl.sh

# 4. verify both variants and (re)generate the report
bash scripts/verify_all.sh
```

Every tool accepts `--category <cat>` and `--case <case_id>` to target a
subset, e.g. `python3 tools/build_cuda.py --category easy --case vectorAdd`.

On Windows PowerShell, use the matching `.ps1` wrappers:

```powershell
.\scripts\setup_env.ps1
.\scripts\run_all_cuda.ps1
.\scripts\run_all_syclomatic.ps1
.\scripts\run_all_sycl.ps1
```

`run_all_cuda.*` and `run_all_sycl.*` also collect a process-level
performance smoke benchmark after correctness verification and write
`reports/performance_status.{csv,md}`. The metric is end-to-end executable
runtime, including program startup and output writing; it is intended as a
repeatable baseline for CUDA-vs-SYCL comparisons, not a pure kernel timer.

## How verification works

Inputs are **deterministic** — either fixed index formulas (`gen_a`/`gen_b`)
or a 32-bit integer hash (`gen_hash01`) reproduced bit-for-bit between the
CUDA/SYCL kernels and numpy float32 (see `tools/verify_lib.py`). Each
`main.cu` writes its
numerical result to `output/<variant>_output.txt`. The case's
`tests/verify.py`:

1. regenerates the same inputs and computes a CPU reference,
2. reads the program output,
3. compares with the metadata tolerance (`max_abs_error`, `max_rel_error`, or
   exact),
4. prints `PASS`/`FAIL` and exits 0/nonzero.

This means the **same** verifier checks both the CUDA and the SYCL output
against an independent CPU reference (spec methods 1–3). Numerical cases use
tolerance, not exact equality.

## How to read the report

`reports/pilot_status.md` has three sections: counts per category, a
pass/fail/skipped tally per pipeline stage, and one row per case. Columns:
`case_id, category, name, cuda_features, libraries, cuda_compile, cuda_run,
cuda_verify, syclomatic_migrate, sycl_compile, sycl_run, sycl_verify,
warnings_count, manual_fixes_required, notes`. `reports/pilot_status.csv` has
the same per-case data for spreadsheets/scripts.

Interpreting a row: `pass` = stage succeeded; `fail` = attempted and failed
(check the matching log under the case's `logs/`); `skipped_*` = a
prerequisite toolchain/device was absent (not a defect of the case).

## Adding a case

1. `cases/<category>/<case_id>/` with the subdir layout above.
2. `original/main.cu` generating deterministic inputs and writing results to
   `argv[1]`; `original/CMakeLists.txt`; `original/README.md`.
3. `tests/verify.py` reusing `tools/verify_lib.py` for the comparison.
4. `metadata.json` following `tools/metadata_schema.json`.
5. `python3 tools/collect_cases.py` to validate, then run the pipeline.
