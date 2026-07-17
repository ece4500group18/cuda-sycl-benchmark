# Stage 2 CUDA-to-SYCL migration benchmark

For a new teammate, harness, or model, start with
[TEAMMATE_RUNBOOK.md](TEAMMATE_RUNBOOK.md). It defines the canonical prompt,
external-harness interface, one-case cost gate, and required result handoff.

Stage 2 evaluates the experimental unit
`case x harness x model x skill_condition x repeat`. A harness (Claude Code,
WorkBuddy, OpenCode, and so on) and a model are separate dimensions. There is
no SYCLomatic baseline and no pre-generated `.dp.cpp`; every real session sees
CUDA `main.cu` and must produce `main.sycl.cpp`.

## What is ready

- The frozen `cuda_verified_250.json` dataset admits only cases whose committed
  NVIDIA build, run, correctness, and performance logs all say `pass`.
- `preflight` rechecks all 1,000 log gates, source/verifier fingerprints, Intel
  Level Zero visibility, an actual SYCL kernel, the harness executable, model
  configuration, and skill files.
- The real baseline is Claude Code with one Opus 4.x model under two conditions:
  native/OOB capability and the same run with `cuda-to-sycl-migration` v1.
- A lower-cost Codex CLI baseline uses `gpt-5.4-mini` at low reasoning effort.
  Codex with a ChatGPT account currently rejects the dated API snapshot, so this
  route records the requested alias, Codex CLI version, run timestamp, detailed
  token classes, and an explicitly labeled API-equivalent cost estimate.
  On this Windows app build, the native workspace-write sandbox resolves no
  writable roots. The Codex config therefore uses its externally-isolated mode:
  the Stage 2 runner first constructs a random whitelist-only temporary
  workspace containing no verifier or repository metadata, then launches Codex
  there and records this mode in raw telemetry.
- Every run records harness/model/skill identity, requested and reported model
  IDs, wall-clock and E2E time, turns, token usage, cost when reported, build,
  run, verification, environment, and the failure funnel.
- JSON, CSV, and Markdown KPI reports refresh automatically after `run`.
- Each model session runs in an ephemeral whitelist-only workspace outside the
  repository tree. Only the resulting sandbox and telemetry are copied into
  artifacts; evaluator metadata and verifier code are never placed there.
- The harness and hidden verifier run locally. Evaluator-owned build/run scripts
  send only `main.sycl.cpp` to the configured Intel SSH worker, return compiler
  feedback to the agent during repair, and download the final output for local
  verification. The server needs neither the repository nor model credentials.

The repository currently contains 292 cases. Only the 250 in the frozen
manifest are NVIDIA-ground-truth complete; the remaining 42 must be run through
Stage 1 on an NVIDIA machine before a new dataset version may include them.

## First Intel GPU run

Follow [INTEL_GPU_RUNBOOK.md](INTEL_GPU_RUNBOOK.md). After the Intel worker is
reachable by SSH key and local Claude Code authentication is available:

```bash
export STAGE2_SSH_TARGET='ubuntu@<intel-host>'
export CLAUDE_BASELINE_MODEL='<exact immutable Opus 4.x model ID>'

python3 tools/stage2/cli.py preflight \
  --experiment benchmark/stage2/experiments/baseline_claude_opus.json \
  --output artifacts/stage2/preflight.json

# Smallest scored smoke: same case, OOB then with-skill.
python3 tools/stage2/cli.py run \
  --experiment benchmark/stage2/experiments/baseline_claude_opus.json \
  --case vectorAdd

# The complete 10-case baseline pilot (20 sessions).
python3 tools/stage2/cli.py run \
  --experiment benchmark/stage2/experiments/baseline_claude_opus.json
```

The model ID environment variable overrides the convenient `opus` alias. Always
set it for scored work so reruns do not silently move to a newer model.

For a low-cost Codex smoke, run one condition first and inspect its measured
token usage before expanding the matrix:

```bash
python3 tools/stage2/cli.py run \
  --experiment benchmark/stage2/experiments/baseline_codex_54mini.json \
  --case vectorAdd --skill oob
```

Codex JSONL reports input, cached input, output, and reasoning output tokens.
The runner does not double-count cached or reasoning tokens. When the ChatGPT
route does not report a dollar charge, `cost_usd` is the API-equivalent estimate
from the price snapshot embedded in the experiment and `cost_source` is
`api_price_estimate`; it is not claimed as an actual subscription invoice.
The current Codex CLI has no per-run hard token-cap flag, so `budget.max_tokens`
is recorded as a post-run alert rather than presented as enforcement. Expand
from one case only after its measured cost is acceptable.

## Offline validation

The mock does not call a model or compile SYCL. It exercises result plumbing and
is always marked `synthetic: true`, `eligible_for_scoring: false`.

```bash
python3 tools/stage2/cli.py plan \
  --experiment benchmark/stage2/experiments/pilot_v1.json
python3 tools/stage2/cli.py run --skip-preflight \
  --experiment benchmark/stage2/experiments/pilot_v1.json \
  --case vectorAdd
python3 -m unittest discover -s tools/stage2/tests -v
```

## Artifacts and reports

```text
artifacts/stage2/<experiment>/<case>/<harness>/<model>/<skill>/repeat-<n>/
  session.json
  harness_stdout.jsonl (or .log)
  harness_stderr.log
  build.json
  run.json
  verify.json
  migration.json
  sandbox/

reports/stage2/<experiment>/
  summary.json
  summary.csv
  summary.md
```

Completed cells are skipped unless `--overwrite` is explicit. Raw artifacts are
ignored by Git. Synthetic results never enter pass-rate, time, token, or cost KPIs.
If only evaluator infrastructure was repaired after a model session, use
`cli.py reevaluate --experiment <config> --result <migration.json>` to repeat
build/run/verification without spending model tokens. The result records
`model_reinvoked: false`.

## Adding another harness or model

Add harnesses and models independently in a schema-v2 experiment. Use a native
adapter when available. For a new CLI, use `external_command` with an `argv`
array. Start from
`benchmark/stage2/templates/external_harness_experiment.json`. Supported
placeholders include `{agent_prompt_file}`, `{prompt}`, `{prompt_file}`,
`{model_id}`, `{sandbox}`, and `{skill_file}`. The CLI must write
`main.sycl.cpp`; it may write a configured telemetry JSON containing
`tokens_in`, `tokens_out`, `tokens_total`, `cost_usd`, `iterations`,
`session_id`, and `model`.

Do not label a candidate model with a marketing family name only. Pin the exact
provider model ID and record the harness version before scored repeats. Compare
OOB and with-skill using identical cases, budgets, device selector, and repeats.

The intended benchmark expansion is harnesses such as Claude Code, Tencent
WorkBuddy, GitHub Copilot, OpenCode, Hermes, Trae, and other reproducible CLIs,
crossed with the strongest available fixed IDs from model families such as
DeepSeek, Qwen, GLM, Kimi, and MiniMax. Candidate names are not activated in a
scored config until their CLI syntax, exact model ID, authentication route,
telemetry fields, and version command have been verified. This avoids recording
a harness marketing name as though it were a model, or silently routing several
nominal models to the same backend.

Global evaluator-owned link flags go in `executor.extra_sycl_flags`; unavoidable
library-specific flags (for example, a frozen oneMKL link choice) can be mapped
by case ID in `executor.case_extra_sycl_flags`. Agents cannot edit these flags:
the evaluator restores the build/run wrappers after every session.

Primary environment and programming references are linked from the Intel runbook
and the migration skill's `references/sources.md`.
