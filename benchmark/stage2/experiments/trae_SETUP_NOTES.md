# TRAE Stage 2 scaffolds

These experiment files mirror the repository's existing Stage 2 pilot format:

- one config per `harness x model` pair;
- the standard 10-case pilot set;
- two skill conditions: `oob` and `with-sycl-skill`;
- the same SSH Intel GPU executor used by the current baselines.

Current limitation:

- this repository has no native `trae` adapter under `tools/stage2/adapters/`;
- no `trae` executable was discoverable on this machine during setup;
- the new TRAE configs therefore use the `external_command` adapter with
  explicit placeholders for the verified headless CLI invocation.

Before a TRAE config can pass `preflight`, replace the harness `argv` template
with the real non-interactive command that:

1. runs inside `{sandbox}`;
2. accepts the selected `{model_id}`;
3. reads the canonical `{agent_prompt_file}`;
4. writes `{sandbox}/main.sycl.cpp`; and
5. preferably writes `stage2_telemetry.json`.

Suggested next commands after filling the real CLI syntax:

```powershell
python tools/stage2/cli.py plan `
  --experiment benchmark/stage2/experiments/trae_<model>.json

python tools/stage2/cli.py preflight `
  --experiment benchmark/stage2/experiments/trae_<model>.json `
  --output artifacts/stage2/preflight-trae-<model>.json

python tools/stage2/cli.py run `
  --experiment benchmark/stage2/experiments/trae_<model>.json `
  --case vectorAdd
```
