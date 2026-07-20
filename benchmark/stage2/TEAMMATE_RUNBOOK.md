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

### Fill in the case list

The template ships `"case_ids": ["vectorAdd"]`. That is a smoke case, not the
matrix. A config copied from the template runs one case in two conditions until
you widen it, and nothing warns you that the other 249 are missing.

Set the list from the manifest rather than typing ids, so a typo cannot silently
shrink the run:

```powershell
# the full scored matrix
python tools/stage2/set_case_ids.py `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json --all

# or a cheaper calibration subset first
python tools/stage2/set_case_ids.py `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json `
  --difficulty hard --dry-run
```

Then confirm the shape before spending anything:

```powershell
python tools/stage2/cli.py plan `
  --experiment benchmark/stage2/experiments/<harness>_<model>.json
```

`plan` prints `runs=<cells> cases=<n>` and the category distribution. With 250
cases, two skill conditions, and one repeat, expect `runs=500 cases=250`.

Keep **one** config per harness/model pair, holding the full case list, for the
whole campaign. Do not create a second config for a subset: a different
`experiment_id` writes to a different artifact directory, so shared cases are
migrated again from scratch and paid for twice. Section 6 explains how to run a
subset out of the full config instead.

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

Omit `--case` only after the projected total cost is accepted.

### Running the full matrix across several sessions

A 500-cell campaign will not fit in one sitting, and no config change is needed
to split it. `--case` is repeatable and filters the configured matrix, so every
batch runs out of the same config and lands in the same artifact directory.

Put the ids for one batch in a file under `benchmark/stage2/batches/`, one per
line, and run it:

```powershell
.\tools\stage2\run_batch.ps1 `
  -Experiment benchmark\stage2\experiments\<harness>_<model>.json `
  -CaseFile benchmark\stage2\batches\<batch>.txt
```

Interrupting with Ctrl+C is safe. `migration.json` is written last, so a cell cut
off mid-flight has no `migration.json`, and the next run redoes it from a clean
sandbox.

What makes batching work is that skipping is per cell, keyed only on whether
`migration.json` exists:

- a case listed in two batches is migrated once; the second batch prints
  `[skipped_existing]` and calls no model, so overlapping batch files cost
  nothing;
- duplicate `--case` values inside one batch collapse, so batch files need no
  deduplication;
- the granularity is (case, harness, model, skill, repeat), so a batch run with
  `--skill oob` still leaves `with-sycl-skill` to run later for the same case;
- `--overwrite` re-executes every selected cell and re-spends the tokens. Never
  pass it to a batch as a matter of routine.

You do not need a final full pass, but one is harmless: running with no `--case`
at the end skips everything finished and picks up only what is missing.

### How the report accumulates

There is one report per `experiment_id`, not one per run. After every `run` the
aggregator rebuilds it by globbing the whole artifact directory, so
`summary.{json,csv,md}` is overwritten each time while its **content is
cumulative** across every batch, including cells finished days earlier. Batching
needs no special handling at report time.

Two consequences worth knowing before you read a summary:

- the aggregator never consults the config, so cells left over from an earlier
  case list still enter the pass rate. Narrowing `case_ids` does not retract
  results already on disk;
- changing budgets or the skill version under an unchanged `experiment_id` mixes
  old and new cells in one report. Bump the `experiment_id` when the conditions
  change.

## 7. Check the run for conduct problems

The scoring pipeline answers whether the SYCL passed, not whether it was
produced the intended way. Audit the transcripts after each batch:

```powershell
python tools/stage2/audit_conduct.py `
  --experiment-id <experiment_id> --quiet --json audit.json
```

It reads artifacts that already exist, so it costs no quota and works on data
collected earlier. It reports:

- `path_escape` — a tool argument outside the session sandbox, which is how an
  agent reading the repository or another case would show up;
- `network` — a shell command that fetches from the network;
- `provenance` — markers in `main.sycl.cpp` that a translation of `main.cu`
  cannot produce, such as the `dpct::` namespace of SYCLomatic output, or a
  licence header or URL absent from the CUDA input;
- `notable_tool` — use of web, skill, or subagent tools. Not misconduct by
  itself; it is a number the report should carry, since it changes what a cell
  measures.

Only the first three set the exit code. Investigate flagged cells individually
before including them in a scored comparison.

State the limit plainly in any writeup: this shows an agent did not *look
something up*, not that it had no prior knowledge. A model reproducing a
translation memorised in training emits no tool call and copies no text, so
neither pass can see it. Ruling that out needs similarity scoring against
published translations, which this repository does not implement.

## 8. Recover from a failed batch

A harness that dies mid-batch — exhausted quota, dropped VPN, expired auth —
does not leave an obvious hole. The runner still writes `migration.json` with
`funnel: "missing"` and `eligible_for_scoring: true`, so the cell is **counted
as a failed migration** and, because `migration.json` now exists, is **skipped**
by the next run. Left alone, an outage silently depresses the pass rate.

Sort the two causes apart before rerunning:

```powershell
python tools/stage2/triage_cells.py --experiment <config.json>
```

A cell is infrastructure-suspect when the session did not report `completed`,
returned a non-zero code, reported no token telemetry, or finished implausibly
fast. Those are the ones an outage produces. A cell where the agent genuinely
ran and produced nothing is left alone, because that is a real result.

Delete the suspect cells so the next run redoes them:

```powershell
python tools/stage2/triage_cells.py --experiment <config.json> --purge
.\tools\stage2\run_batch.ps1 -Experiment <config.json> -CaseFile <batch>.txt
```

Passing `--experiment` also cross-checks the artifacts against the configured
matrix and lists which cells are still pending, which is the reliable way to see
campaign progress. Cells that failed with `compile_error`, `run_error`, or
`wrong_output` are never purged — those are valid results and rerunning them
would bias the comparison toward whichever model got retried.

If the model already produced `main.sycl.cpp` and only the evaluator broke, use
`cli.py reevaluate` from section 5 instead; it repeats build, run, and
verification without spending another session.

## 9. Return these results

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
