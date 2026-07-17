# Intel GPU execution runbook

The harness/model, frozen dataset, hidden verifier, KPI collection, and reports
stay on the local controller. The Intel server is an SSH execution worker: it
receives only the current `main.sycl.cpp`, compiles and runs it under `/tmp`,
and returns program output and command logs.

The Intel server does not need Claude Code, model credentials, the benchmark
repository, case metadata, or verifier code. The evaluator is intentionally
single-GPU/sequential.

## 1. Install and expose the Intel GPU stack

Use Intel's Linux GPU installation instructions for the target distribution:
https://dgpu-docs.intel.com/overview/introduction.html

The required path is the Intel kernel-mode driver (`xe` or `i915` as appropriate),
Level Zero user-mode runtime, oneAPI DPC++ compiler, and permission to a
`/dev/dri/renderD*` node. Install oneAPI Base Toolkit or an equivalent DPC++
toolchain, then initialize its environment for each shell, commonly:

```bash
source /opt/intel/oneapi/setvars.sh
```

On the Intel server, these should succeed:

```bash
icpx --version
sycl-ls --ignore-device-selectors
```

The local `preflight` later repeats device discovery through SSH, transfers a
small USM `single_task`, compiles it remotely, executes it on Level Zero, and
downloads its output. Fix render-node group membership or device passthrough if
the SSH user cannot read and write `/dev/dri/renderD*`.

## 2. Prepare non-interactive SSH locally

Use an SSH key or an organization-managed SSH agent. Never store the server
password in an experiment JSON, command line, report, or model-visible
environment variable.

```bash
ssh-copy-id ubuntu@<intel-host>
ssh -o BatchMode=yes ubuntu@<intel-host> true
export STAGE2_SSH_TARGET='ubuntu@<intel-host>'
```

The experiment reads only `STAGE2_SSH_TARGET`. Its default SSH/SCP options use
`BatchMode=yes`, so missing key authorization fails immediately instead of
hanging a model session at a password prompt.

## 3. Prepare the local harness and model

Install and authenticate Claude Code on the local controller using Anthropic's
official setup guide:
https://docs.anthropic.com/en/docs/claude-code/getting-started

```bash
claude --version
export CLAUDE_BASELINE_MODEL='<exact immutable Opus 4.x model ID>'
```

The experiment also works with an Anthropic-compatible gateway configured for
Claude Code, but gateway URL, authentication, and model routing are deployment
credentials and intentionally are not committed. Official gateway variables are
documented at https://docs.anthropic.com/en/docs/claude-code/llm-gateway.

## 4. Run the hard gate locally

```bash
python3 tools/stage2/cli.py preflight \
  --experiment benchmark/stage2/experiments/baseline_claude_opus.json \
  --output artifacts/stage2/preflight.json
```

Do not use `--skip-preflight` for scored work. `--skip-preflight` exists only for
offline mock tests and controlled failure diagnostics.

## 5. Produce the first report locally

Run one representative case first. This creates one OOB session and one
with-skill session and then writes the report automatically.

```bash
python3 tools/stage2/cli.py run \
  --experiment benchmark/stage2/experiments/baseline_claude_opus.json \
  --case vectorAdd
```

Inspect:

```text
reports/stage2/baseline-claude-opus-pilot-v1/summary.md
reports/stage2/baseline-claude-opus-pilot-v1/summary.csv
reports/stage2/baseline-claude-opus-pilot-v1/summary.json
```

Then run all 10 pilot cases by omitting `--case`. Increase `repeats` only after
the pilot is stable and the exact model ID, harness version, skill version,
budget, and device are frozen.

## 6. Admit the remaining CUDA cases only after NVIDIA validation

On an NVIDIA machine, run the repository's Stage 1 build, run, verify, and
benchmark commands for every pending case. Confirm strict metadata and reports,
then create a new immutable manifest; do not overwrite the 250-case manifest.

```bash
python3 tools/run_all.py --stage cuda_verify
python3 tools/run_all.py --stage cuda_benchmark
python3 tools/check_metadata.py --strict-stage1
python3 tools/stage2/cli.py manifest \
  --output benchmark/stage2/datasets/cuda_verified_<new-count>_v2.json \
  --expected-count <new-count> \
  --dataset-id cuda-verified-<new-count>-v2
```

Review the new manifest diff and point a new experiment ID at it. Existing
results remain bound to the original dataset ID, commit, and fingerprints.

## Primary programming references

- Intel oneAPI Programming Guide 2025.1:
  https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2025-1/overview.html
- Khronos SYCL 2020 specification:
  https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
