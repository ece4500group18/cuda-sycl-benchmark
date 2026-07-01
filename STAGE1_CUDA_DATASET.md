# Stage 1 CUDA Dataset Validation

This repository stage is CUDA ground-truth validation on NVIDIA GPUs. It does
not evaluate SYCL migration, Intel GPU execution, or SYCLomatic as a baseline.

## Current Scope

The root `tools/` commands discover both:

- `pilot_benchmark/cases/*/*`
- `benchmark/collection/*/cases/*`

Each case is treated as an original CUDA benchmark with metadata, source,
build/run commands, verifier, and logs.

## Current Snapshot

Snapshot date: 2026-07-01 Asia/Shanghai.

- Total cases: 130
- Actual CUDA verification: 130/130 pass
- NVIDIA performance baseline: 130/130 pass
- Strict Stage 1 metadata validation: 130/130 pass
- Stage 2 status: not evaluated

The latest expansion added 38 cases, all as real-project simplified
extractions/adaptations:

- ggml-org/llama.cpp: 10 cases
- vllm-project/vllm: 8 cases
- bitsandbytes-foundation/bitsandbytes: 6 cases
- Dao-AILab/flash-attention: 4 cases
- facebookresearch/xformers: 2 cases
- ORNL/HeCBench: 8 cases

Overall source composition:

- Real-project extracted/adapted cases: 72
- Hand-written repository-authored cases: 58

Current distributions:

- Domain: cuda_primitive 20, hpc 21, image_processing 20, library_api 10, modern_ml 59
- Difficulty: easy 12, medium 40, hard 78
- License: MIT 74, Apache-2.0 40, BSD-3-Clause 14, BSD-style / BSD-3-like 2

## Commands

```bash
python tools/check_metadata.py
python tools/collect_cases.py
python tools/report_status.py
python tools/migrate_metadata_stage1.py
python tools/check_metadata.py --strict-stage1

python tools/build_case.py --case <case_name>
python tools/run_case.py --case <case_name>
python tools/verify_case.py --case <case_name>
python tools/benchmark_case.py --case <case_name>

python tools/run_all.py --stage cuda_verify
python tools/run_all.py --stage cuda_benchmark
```

Use `--case <case_name>` to target one case. Omit it to process every case.

## Logs

Stage 1 logs are written under each case directory:

- `logs/build_result.json`
- `logs/run_result.json`
- `logs/verify_result.json`
- `logs/perf_result.json`

Raw command output is written to `logs/cuda_compile.log`, `logs/cuda_run.log`,
`logs/verify.log`, and per-measurement performance logs.

## Reports

Root-level reports are generated under `reports/`:

- `reports/collection_audit.json`
- `reports/metadata_validation.json`
- `reports/dataset_summary.json`
- `reports/dataset_summary.csv`
- `reports/dataset_summary.md`

`reports/dataset_summary.md` contains the dataset summary table, counts by
domain/difficulty/status/source project/license, build-ready and verify-ready
lists, NVIDIA-verified lists, performance-recorded lists, real-project and
hand-written case lists, the Real Project Extraction Summary, and Stage 2
remaining work.

The report distinguishes:

- `declared_status`: the legacy status summarized from `metadata.json`
- `actual_verify_status`: status from `logs/verify_result.json`
- `actual_perf_status`: status from `logs/perf_result.json`

Only actual log statuses count as verified or performance-ready.

The Real Project Extraction Summary groups extracted/adapted cases by source
project and records representative kernels, license, extraction notes,
verification method, and whether benchmark logs passed.

## Metadata Migration

Run:

```bash
python tools/migrate_metadata_stage1.py
python tools/check_metadata.py --strict-stage1
```

The migration writes `metadata.stage1.json` sidecars and leaves legacy
`metadata.json` untouched so the older pilot tools keep working.

## Stage 2 Boundary

Stage 2 is intentionally out of scope here. Later work can add migrated-code
metadata for agent name, compile success, correctness, runtime, CUDA-vs-SYCL
performance ratio, token use, repair attempts, and migration time.
