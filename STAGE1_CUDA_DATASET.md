# Stage 1 CUDA Dataset Validation

This repository stage is CUDA ground-truth validation on NVIDIA GPUs. It does
not evaluate SYCL migration, Intel GPU execution, or SYCLomatic as a baseline.

## Current Scope

All cases live in a single unified layout discovered by the root `tools/`
commands:

- `benchmark/cases/<category>/<case>/`

`<category>` is one of the ten collection category slugs defined in
`benchmark/collection/README.md` (simple-kernels, memory-movement,
stencil-convolution, reductions-scans, graph-irregular, molecular-dynamics,
linear-algebra, multi-kernel-pipelines, cuda-library-usage,
streams-atomics-templates). Domain and difficulty are metadata fields, not
directory levels. The legacy `pilot_benchmark/cases/*/*` and
`benchmark/collection/*/cases/*` layouts were consolidated into this tree
on 2026-07-03 (git history preserves the renames).

Each case is treated as an original CUDA benchmark with metadata, source,
build/run commands, verifier, and logs.

## Current Snapshot

Snapshot date: 2026-07-01 Asia/Shanghai (contents unchanged by the
2026-07-03 layout consolidation; all 250 cases re-validated with strict
metadata checks after the move).

- Total cases: 250
- Actual CUDA build: 250/250 pass
- Actual CUDA run: 250/250 pass
- Actual CUDA verification: 250/250 pass
- NVIDIA performance baseline: 250/250 pass
- Strict Stage 1 metadata validation: 250/250 pass
- Stage 2 status: not evaluated

The expansion from the stable 130-case checkpoint added 120 cases in three
verified Stage 1 batches:

- Batch 1: 41 cases, 41/41 build/run/verify/benchmark pass
- Batch 2: 40 cases, 40/40 build/run/verify/benchmark pass
- Batch 3: 39 cases, 39/39 build/run/verify/benchmark pass

Overall source composition:

- External real-project or benchmark-suite sourced cases: 192
- Real-project extracted/adapted cases reported by extraction summary: 132
- Hand-written repository-authored cases: 58

Current distributions:

- Domain: cuda_primitive 40, hpc 51, image_processing 35, library_api 20, modern_ml 104
- Difficulty: easy 18, medium 137, hard 95
- License: MIT 94, Apache-2.0 65, BSD-3-Clause 59, BSD-3-Clause + CUDA EULA note 30, BSD-style / BSD-3-like 2

Current external source distribution:

- ORNL/HeCBench: 38 cases
- NVIDIA/DALI: 35 cases
- NVIDIA/cuda-samples: 30 cases
- ggml-org/llama.cpp: 24 cases
- vllm-project/vllm: 18 cases
- bitsandbytes-foundation/bitsandbytes: 12 cases
- open-mmlab/mmcv: 12 cases
- Dao-AILab/flash-attention: 10 cases
- NVIDIA/cutlass: 7 cases
- facebookresearch/xformers: 4 cases
- pytorch/pytorch: 2 cases

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
project and records representative kernels, license, extraction fidelity,
extraction notes, verification method, and whether benchmark logs passed.

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
