# Stage 2 hard-50 campaign results: baseline-claude-opus47-local-v1

Campaign window: 2026-07-30 to 2026-08-03 (two sittings, split by subscription
session-limit resets). This document explains the scored results; the
machine-generated cumulative report is `summary.{md,csv,json}` in this
directory.

## Experiment setup

| dimension | value |
| --- | --- |
| harness | Claude Code CLI 2.1.170, adapter `claude_code`, permission mode `acceptEdits` |
| model | `claude-opus-4-7` (immutable ID, provider-reported telemetry) |
| skill conditions | `oob` (no skill) and `with-sycl-skill` (`benchmark/stage2/skills/cuda-to-sycl-migration` v1) |
| case set | the 50 hard cases from `benchmark/stage2/batches/hard10.txt` + `hard40.txt`, a subset of the frozen manifest `cuda-verified-250-v1` (commit `5189cbc`) |
| matrix | 50 cases x 2 conditions x 1 repeat = 100 cells |
| budget per cell | 500k tokens (post-run observation), 30 iterations, 3600 s wall clock |
| executor | local, single GPU, sequential |
| device | Intel UHD Graphics 750 (RocketLake iGPU, 32 EU), `ONEAPI_DEVICE_SELECTOR=opencl:gpu`, NEO driver 23.43.027642 |
| compiler | oneAPI icpx 2026.0, `-fsycl` |
| prompt | `translation-task-v2`, assembled verbatim from `benchmark/stage2/prompts/` |

Preflight (`cli.py preflight`, no `--skip-preflight`) passed all five gates
(dataset fingerprint, harness, model, skill, Intel GPU live-kernel probe)
before each sitting.

## Headline results (hard-50 only, 100 cells)

| condition | scored | passed | pass rate | mean E2E | mean tokens/cell | total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oob | 50 | 44 | **0.88** | 211 s | 1.98M | $71.00 |
| with-sycl-skill | 50 | 45 | **0.90** | 207 s | 2.02M | $72.39 |
| combined | 100 | 89 | 0.89 | 209 s | 2.00M | $143.39 |

The skill condition passed one more case than OOB (daliWarpAffine failed only
in OOB). A one-case difference on a 50-case set is not statistically
meaningful; the practical takeaway is that Opus 4.7 translates ~90% of hard
CUDA kernels to functionally correct SYCL on the first session, with or
without the migration skill.

Token/cost figures are provider-reported per session. Mean tokens (~2M/cell)
are dominated by cache reads; billed cost averaged ~$1.43 per cell.

## Failure analysis (11 failed cells, 6 distinct cases)

Five cases failed in **both** conditions, all `wrong_output` (compiled and ran,
hidden verifier rejected the numerics):

| case | category | failure mode |
| --- | --- | --- |
| bnbAdam8bitMoments | simple-kernels | 8-bit Adam moment quantization semantics (max_abs_error > 1e-5) |
| bnbInt8VectorQuant | reductions-scans | int8 vector quantization scaling |
| ggmlQuantizeQ8 | reductions-scans | Q8 blockwise quantization |
| ggmlRopeInterleaved | reductions-scans | RoPE with interleaved pair layout |
| vllmRotaryPaged | graph-irregular | RoPE applied through paged KV indirection |

One case failed only in OOB: **daliWarpAffine** (`run_error`) — the delivered
`main.sycl.cpp` still contained debugging scaffolding (prints
`start` / `queue ok` then exits before the kernel, never writes the output
file). The with-skill session passed the same case.

Two clear failure clusters:

1. **Quantization semantics** (4 of 6 cases): low-bit quantize/dequantize
   kernels where the translation compiles and runs but drifts numerically.
   Absmax/scale rounding and per-block layout details are where the model's
   CUDA-to-SYCL mapping breaks.
2. **RoPE layouts** (2 of 6 cases): interleaved/paged rotary embedding
   indexing.

A third mode — **debug scaffolding left in the final deliverable** (the agent
bisects a device issue with probe prints/early exits and fails to restore the
real implementation before the session ends) — produced daliWarpAffine's OOB
failure here and was also the cause of all 3 pilot failures (nbodyTiled
with-skill, thrustSort both conditions). A cheap "does the deliverable still
write the output file" self-check in the prompt or harness would likely
recover most of these.

## Infrastructure incidents (recovered per runbook §8)

The campaign hit the Claude subscription session limit twice (2026-07-30
~16:50 SGT and again in the second sitting). Each time the harness died
instantly for every remaining cell, the runner recorded `funnel: missing`
cells (0 tokens, ~2 s, returncode 1), and the batch "completed" with a
depressed pass rate. Recovery followed the runbook: `triage_cells.py`
identified the infra-suspect cells (80, then 38), `--purge` deleted exactly
those cells, and the run was re-executed after the quota reset. Genuine
failures (`wrong_output` / `run_error`) were never purged or rerun. Final
triage: 0 infra-suspect cells; artifacts cover the configured matrix exactly.

## Conduct audit

`audit_conduct.py` over all 118 cells: **0 network flags, 0 provenance
flags**. 45 cells carried 185 `path_escape` flags; every flagged argument was
inspected and all are benign scratch I/O in `/tmp` (`/tmp/test.txt`,
`/tmp/out.txt`, probe programs like `/tmp/probe.cpp`, build logs) plus two
references to the session's own `/tmp/stage2-*/workspace` sandbox by absolute
path and one `/usr/bin/env bash` invocation. No agent read the repository,
another case, the verifier, or the network. All cells are scoreable. Standard
limitation: this shows the agents did not *look anything up*, not that the
model had no memorized prior knowledge of these kernels.

## Reading `summary.md` vs this document

`summary.{md,csv,json}` is rebuilt by the aggregator from the whole artifact
directory, which also contains 18 cells from the earlier 10-case pilot
(9 non-hard cases; `attention` overlaps and is counted once). Its headline
numbers (118 scored, 0.881 overall) therefore mix pilot and hard-50 cells.
The hard-50-only numbers in this document were filtered by the frozen
hard10+hard40 case list. Pilot-only results: 14/18 passed; failures were
nbodyTiled with-skill and thrustSort both conditions (debug-scaffold mode) —
see the git history of this directory for the pilot-era summary.

## Reproduction

```bash
source /opt/intel/oneapi/setvars.sh
export CLAUDE_BASELINE_MODEL=claude-opus-4-7
python3 tools/stage2/cli.py preflight \
  --experiment benchmark/stage2/experiments/baseline_claude_opus47_local.json \
  --output artifacts/stage2/preflight.json
python3 tools/stage2/cli.py run \
  --experiment benchmark/stage2/experiments/baseline_claude_opus47_local.json
python3 tools/stage2/audit_conduct.py \
  --experiment-id baseline-claude-opus47-local-v1 --quiet --json audit.json
```

Raw per-cell artifacts (`migration.json`, `session.json`, transcripts,
`main.sycl.cpp`, build/run/verify logs) live under
`artifacts/stage2/baseline-claude-opus47-local-v1/` (git-ignored). The
preflight and audit JSONs for this campaign are committed alongside this
document as `preflight-hard50.json` and `audit-hard50.json`.
