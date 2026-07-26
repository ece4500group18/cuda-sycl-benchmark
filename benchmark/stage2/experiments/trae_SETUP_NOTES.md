# TRAE Stage 2 scaffolds

These experiment files mirror the repository's existing Stage 2 pilot format:

- one config per `harness x model` pair;
- the standard 10-case pilot set;
- two skill conditions: `oob` and `with-sycl-skill`;
- the same SSH Intel GPU executor used by the current baselines.

Current status:

- this repository has no native `trae` adapter under `tools/stage2/adapters/`;
- the experiment configs now use the verified external command:
  `E:\SJTU Courses\Senior Summer\capstone\trae-agent\.venv\Scripts\python.exe`
  plus `tools/stage2/trae_openrouter_wrapper.py`, which calls
  `trae-cli.exe`, writes `stage2_trajectory.json`, and synthesizes
  `stage2_telemetry.json`;
- the command points at the local git-ignored OpenRouter config:
  `.local/trae_config.openrouter.yaml`;
- exact OpenRouter routes are currently pinned for:
  `deepseek/deepseek-v4-pro`, `minimax/minimax-m3`,
  `z-ai/glm-5.2`, and `moonshotai/kimi-k3`;
- `gpt` and `opus` still need exact route names before scored runs.

Current limitation:

- the wrapper now emits token telemetry from the Trae trajectory, including
  `tokens_in`, `tokens_out`, `cached_input_tokens`,
  `reasoning_output_tokens`, `tokens_total`, `iterations`, and `model`;
- provider-reported `cost_usd` and `session_id` are still unavailable from the
  current Trae CLI/OpenRouter path and will remain `null` unless a deeper
  integration is added.

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
