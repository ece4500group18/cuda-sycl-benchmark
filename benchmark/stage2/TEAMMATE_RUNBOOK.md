# Teammate runbook: harness x model migration experiments

This runbook is the handoff contract for testing a new agent harness and model
pair. Do not change the CUDA inputs, canonical prompt, migration skill, Intel
device selector, verifier, or budget while comparing pairs.

## 1. Experimental unit

One scored unit is:

```text
case x harness x model x skill_condition x repeat
```

A harness is the agent application or CLI, such as Claude Code, Codex CLI,
OpenCode, WorkBuddy, or Trae. A model is the exact backend model ID routed by
that harness. Record them as separate dimensions.

Use a unique experiment ID containing both names, for example:

```text
pilot-opencode-qwen3.5-exactid-v1
```

Do not use a marketing family name when an immutable provider model ID is
available. If the route exposes only an alias, record the alias, harness
version, provider route, and run timestamp and label that limitation.

## 2. Local prerequisites

The harness and model credentials stay on your own computer. The Intel server
does not need the harness, repository, or credentials.

Required locally:

- this repository and team branch;
- Python with NumPy available;
- Git Bash on Windows (WSL bash uses a different SSH home);
- your assigned harness installed and authenticated;
- non-interactive SSH access to the Intel worker.

Ask the server owner to authorize your own public key. Never commit or share a
private key, password, API key, or access token.

PowerShell:

```powershell
$env:STAGE2_SSH_TARGET = "ubuntu@<intel-host>"
ssh -o BatchMode=yes $env:STAGE2_SSH_TARGET true
python -c "import numpy; print(numpy.__version__)"
```

Bash:

```bash
export STAGE2_SSH_TARGET='ubuntu@<intel-host>'
ssh -o BatchMode=yes "$STAGE2_SSH_TARGET" true
python3 -c 'import numpy; print(numpy.__version__)'
```

Only one teammate should use the Intel GPU at a time. The runner is sequential
on one controller, but it does not implement a distributed lock across laptops.

## 3. What the model can see

Each session receives a random, whitelist-only workspace containing:

```text
AGENT_PROMPT.md
TASK.md
main.cu
CMakeLists.txt
sycl_build.sh
sycl_run.sh
remote_exec.py
remote_config.json
output/
```

The model cannot see the verifier, tolerances, reference output, Stage 1 logs,
other cases, or another model's migration. The with-skill condition additionally
receives `skill/`; the OOB condition does not.

The canonical prompt is assembled verbatim from:

- `benchmark/stage2/prompts/base.txt`;
- `benchmark/stage2/prompts/oob.txt`, or
- `benchmark/stage2/prompts/with_skill.txt`; and
- `benchmark/stage2/prompts/finish.txt`.

The detailed migration and output contract is
`benchmark/stage2/TRANSLATION_TASK.md`, copied into the workspace as `TASK.md`.
Every adapter writes the exact combined text to `agent_prompt.txt` in the raw
run artifacts.

## 4. Add your harness and model

Native adapters currently exist for Codex CLI and Claude Code. For another CLI,
copy:

```text
benchmark/stage2/templates/external_harness_experiment.json
```

to:

```text
benchmark/stage2/experiments/<harness>_<model>.json
```

Replace every `REPLACE_...` value and verify the real non-interactive CLI syntax
from that harness's documentation. Do not guess flags.

The `external_command` adapter supports these literal placeholders:

| Placeholder | Value |
| --- | --- |
| `{model_id}` | resolved exact model ID |
| `{sandbox}` | isolated current-case workspace |
| `{agent_prompt_file}` | canonical combined prompt file |
| `{prompt}` | canonical combined prompt text as one argument |
| `{prompt_file}` | detailed `TASK.md` path |
| `{skill_file}` | `skill/SKILL.md`, or empty for OOB |

The harness must run with `{sandbox}` as its effective working directory, obey
the canonical prompt, and produce `{sandbox}/main.sycl.cpp`.

If the CLI can emit telemetry, write `stage2_telemetry.json` inside the sandbox:

```json
{
  "tokens_in": 1000,
  "tokens_out": 200,
  "tokens_total": 1200,
  "cost_usd": 0.0123,
  "iterations": 4,
  "session_id": "provider-session-id",
  "model": "exact-reported-model-id",
  "message": "completed"
}
```

Leave unknown fields absent rather than inventing zero values. `cost_usd` should
be provider-reported actual cost when available. If only token counts and a
public price table are available, add a pricing snapshot to the model block so
the runner can label the result as an estimate.

If the CLI cannot be expressed by an argv template or its native JSONL needs
special parsing, add a native adapter under `tools/stage2/adapters/`, register it
in `runner.create_adapter`, add it to the experiment schema, and add unit tests.

## 5. Run the gate and one-case calibration

Never use `--skip-preflight` for scored work.

```powershell
python tools/stage2/cli.py plan `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json

python tools/stage2/cli.py preflight `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json `
  --output artifacts/stage2/preflight-<harness>-<model>.json

python tools/stage2/cli.py run `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json `
  --case vectorAdd
```

The last command runs both OOB and with-skill. Inspect token usage, cost, model
identity, build, run, correctness, and failure funnel before adding more cases.
The current `max_tokens` field is a post-run alert unless the harness itself
offers a verified hard token cap.

If only evaluator infrastructure was broken after a model already produced
`main.sycl.cpp`, use `cli.py reevaluate`; do not spend another model session:

```powershell
python tools/stage2/cli.py reevaluate `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json `
  --result artifacts/stage2/<experiment>/<case>/<harness>__<model>__<skill>__r0/migration.json
```

## 6. Expand only after calibration

Use the same frozen manifest, case list, prompt version, skill version, budgets,
device selector, and repeat count for every compared pair.

Run additional cases individually first:

```powershell
python tools/stage2/cli.py run `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json `
  --case transposeShared
```

Omit `--case` only after the projected total cost is accepted. Completed cells
are skipped automatically unless `--overwrite` is explicitly requested.

## 7. Return these results

Commit the experiment JSON and generated report:

```text
reports/stage2/<experiment>/summary.md
reports/stage2/<experiment>/summary.csv
reports/stage2/<experiment>/summary.json
```

Raw `artifacts/stage2/` are ignored by Git. Review them for secrets, then share a
separate archive containing, for every scored cell:

```text
migration.json
session.json
agent_prompt.txt
harness_stdout.jsonl or harness_stdout.log
harness_stderr.log
build.json
run.json
verify.json
sandbox/main.sycl.cpp
```

Also return the preflight JSON and the output of the harness's version command.
Do not include credentials, private keys, or provider authentication files.
