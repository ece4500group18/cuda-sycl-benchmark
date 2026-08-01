# TRAE hard50 report

## Scope

This report summarizes the **hard50-v1** results produced by the Stage 2 **TRAE** harness (`external_command` adapter + `trae-cli`) on the `cuda-verified-250-v1` dataset (hard50 subset). It focuses on two prompting conditions per model: `oob` and `with-sycl-skill`.

At the time of writing, the repository contains completed hard50 summaries for:

- `trae-deepseek-v4pro-hard50-v1`
- `trae-glm-5.2-hard50-v1`

Hard50 summaries for `Kimi` and `Minimax` are not present under `reports/stage2/` yet, so this document only reports DeepSeek and GLM.

## Overall results (hard50)

| Model | Experiment | Total | Pass | Wrong output | Compile error | Missing | Pass rate | Total tokens | Avg tokens / cell |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Pro | `trae-deepseek-v4pro-hard50-v1` | 100 | 60 | 10 | 19 | 11 | 60% | 38,347,823 | 383,478 |
| GLM 5.2 | `trae-glm-5.2-hard50-v1` | 100 | 60 | 10 | 5 | 25 | 60% | 25,302,597 | 253,026 |

Notes:

- Both models end up at the same **pass rate (60%)** on hard50.
- DeepSeek uses substantially more tokens overall, but also has fewer `missing` than GLM.
- GLM has a much higher `missing` rate, but far fewer `compile_error` than DeepSeek.

## Results by condition

### DeepSeek V4 Pro

| Condition | Attempts | Pass | Wrong output | Compile error | Missing | Pass rate | Mean e2e (s) | Median e2e (s) | Mean tokens | Mean iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oob` | 50 | 30 | 5 | 11 | 4 | 60% | 1748.9 | 453.4 | 394,031 | 26.28 |
| `with-sycl-skill` | 50 | 30 | 5 | 8 | 7 | 60% | 597.3 | 479.8 | 372,925 | 26.62 |

Interpretation:

- Skill conditioning keeps pass rate unchanged, but shifts failure modes: fewer `compile_error` (11 → 8) and more `missing` (4 → 7).
- DeepSeek `oob` has a very large mean e2e due to a few extreme outliers (median e2e is in the same range as with-sycl-skill).

### GLM 5.2

| Condition | Attempts | Pass | Wrong output | Compile error | Missing | Pass rate | Mean e2e (s) | Median e2e (s) | Mean tokens | Mean iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `oob` | 50 | 30 | 5 | 1 | 14 | 60% | 335.2 | 309.0 | 203,048 | 21.86 |
| `with-sycl-skill` | 50 | 30 | 5 | 4 | 11 | 60% | 815.9 | 279.4 | 303,004 | 24.88 |

Interpretation:

- Skill conditioning reduces `missing` (14 → 11) but increases `compile_error` (1 → 4).
- Mean tokens and mean e2e increase notably with the skill, suggesting longer/less stable agent runs on some hard cases.

## Remaining missing cases (after reruns)

Across both models, all remaining `missing` cells share the same immediate scorer-side cause:

- `build.message = "adapter did not produce main.sycl.cpp"`
- `raw_telemetry.returncode = 0`

So these are not “harness crashed” outcomes; they are cases where the agent run completed, but did not leave a `main.sycl.cpp` artifact for Stage 2 to build/run/verify.

### DeepSeek V4 Pro (11 missing)

- `oob` (4): `daliNormalizePermute`, `daliRandomResizedCrop`, `mmcvBallQuery`, `mmcvGatherPoints`
- `with-sycl-skill` (7): `daliRandomResizedCrop`, `daliResizeCropMirror`, `mmcvGroupPoints`, `mmcvKnn`, `mmcvRoiAlign`, `mmcvRoiPool`, `pytorchMaxPool2d`

### GLM 5.2 (25 missing)

- `oob` (14): `daliNormalizePermute`, `daliRandomResizedCrop`, `daliResizeCropMirror`, `mmcvBallQuery`, `mmcvBboxOverlaps`, `mmcvFurthestPointSample`, `mmcvGatherPoints`, `mmcvGroupPoints`, `mmcvKnn`, `mmcvNms`, `mmcvPointsInBoxes`, `mmcvRoiAlign`, `mmcvRoiPool`, `pytorchMaxPool2d`
- `with-sycl-skill` (11): `daliNormalizePermute`, `daliRandomResizedCrop`, `daliResizeCropMirror`, `mmcvBallQuery`, `mmcvBboxOverlaps`, `mmcvFurthestPointSample`, `mmcvGatherPoints`, `mmcvPointsInBoxes`, `mmcvRoiAlign`, `mmcvRoiPool`, `pytorchMaxPool2d`

### Missing failure subtypes

Even though the scorer labels all of them as `missing`, the logs suggest at least three distinct subtypes:

1. **Early exit / agent-side tooling failure (very low steps and tokens)**
   - Example: `GLM / oob / mmcvNms` ends in ~8 seconds (`iterations=2`, `tokens_total=6086`) with an agent-side parse error (`Expecting value...`) recorded in `trae_cli_stdout.log`.
   - These look less like “hard task failure” and more like agent workflow/tooling brittleness.

2. **Budget exhaustion without artifact**
   - Example: `GLM / with-sycl-skill / pytorchMaxPool2d` hits `iterations=30` with `tokens_total=726,785` and message `Task execution exceeded maximum steps without completion.`
   - These look like genuine “hard case” failures: the agent is engaged but does not converge to a valid migration artifact.

3. **Mid-run incompletion (moderate steps/tokens, still no `main.sycl.cpp`)**
   - Many of the remaining `DALI` and `MMCV` cases fall into this category (often 6–17 iterations with tens to hundreds of thousands of tokens).
   - These are likely a mix of “did not reach the file-write step” and “wrote partial changes but never produced the required artifact”.

## Conclusions

1. **DeepSeek vs GLM on hard50**
   - Both end at **60% pass rate**, but with different failure profiles.
   - DeepSeek: fewer `missing`, more `compile_error`, higher token usage.
   - GLM: many more `missing`, very few `compile_error`, lower token usage overall.

2. **Skill impact is model-dependent**
   - DeepSeek: skill reduces compile errors but increases missing; pass rate unchanged.
   - GLM: skill reduces missing but increases compile errors and cost (tokens/e2e); pass rate unchanged.

3. **Remaining missing cases are not primarily “network missing”**
   - With VPN stability confirmed, the remaining missing cases overwhelmingly reflect “agent completed without producing `main.sycl.cpp`”.
   - A minority are clearly agent-side early-exit/tooling failures, which may be worth targeted reruns or improving the wrapper/tool interaction.

## Recommended next steps

1. For reporting, treat `missing` as “no migration artifact” and optionally sub-label:
   - `missing_early_exit`
   - `missing_budget_exhausted`
   - `missing_incomplete_artifact`

2. If you want to try to reduce `missing` further without changing model prompts:
   - Prioritize reruns of the **early-exit** cases (low iters/tokens), because they are more likely to be stochastic/tooling-related.
   - Do not over-invest reruns on repeated budget-exhaustion cases; they are likely true hard failures.

3. Produce and commit hard50 summaries for `Kimi` and `Minimax` so the four-model comparison is complete under the same hard50 benchmark.
