# TRAE Pilot Report

## Scope

This note summarizes the 10-case pilot results for four models evaluated under the TRAE framework:

The original 10 pilot cases were: `vectorAdd`, `transposeShared`, `conv1dShared`, `reduceSum`, `bfs`, `nbodyTiled`, `tiledMatmul`, `attention`, `thrustSort`, and `histogram`.

- `DeepSeek V4 Pro`
- `Kimi K3`
- `GLM 5.2`
- `Minimax M3`

Each model was tested under two prompting conditions:

- `oob` (`out-of-box`)
- `with-sycl-skill`

That gives `20` scored cells per model.

This report is intended as an intermediate analysis only. It is useful for identifying current strengths, weaknesses, and likely failure modes, but it is not a substitute for the final large-scale analysis after the broader benchmark finishes.

## Data basis

The conclusions below are based on the final `migration.json` results stored under:

- `reports/stage2/trae-minimax-m3-pilot-v1/summary.json`
- `reports/stage2/trae-deepseek-v4pro-pilot-v1/summary.json`
- `reports/stage2/trae-glm-5.2-pilot-v1/summary.json`
- `reports/stage2/trae-kimi-v3-pilot-v1/summary.json`

For `GLM` and `Kimi`, some cells were initially marked as `harness_error` because of Windows file-lock issues (`WinError 32`). Those were rerun successfully. The analysis in this report uses the final scored `migration.json` outcomes rather than the transient harness-side failures.

## Overall ranking

By pilot pass count, the current ranking is:

| Model | Pass | Total | Pass rate |
|---|---:|---:|---:|
| DeepSeek V4 Pro | 20 | 20 | 100% |
| Kimi K3 | 17 | 20 | 85% |
| GLM 5.2 | 16 | 20 | 80% |
| Minimax M3 | 13 | 20 | 65% |

At this stage, `DeepSeek V4 Pro` is the strongest model in the TRAE setup by a clear margin. It is the only model that passed every pilot cell under both prompting conditions.

## Condition-level summary

### Out-of-box

| Model | Pass | Total | Pass rate |
|---|---:|---:|---:|
| DeepSeek V4 Pro | 10 | 10 | 100% |
| Kimi K3 | 9 | 10 | 90% |
| Minimax M3 | 7 | 10 | 70% |
| GLM 5.2 | 7 | 10 | 70% |

### With SYCL skill

| Model | Pass | Total | Pass rate |
|---|---:|---:|---:|
| DeepSeek V4 Pro | 10 | 10 | 100% |
| GLM 5.2 | 9 | 10 | 90% |
| Kimi K3 | 8 | 10 | 80% |
| Minimax M3 | 6 | 10 | 60% |

## Main takeaways

### DeepSeek V4 Pro

`DeepSeek V4 Pro` is currently the strongest and most reliable model in the TRAE framework. It achieved `20/20` passes, with no `missing`, `wrong_output`, or `compile_error` cells in the pilot.

Its main advantage is robustness across case types rather than just speed or token efficiency. It handled all tested categories, including hard cases, without changing the failure mode profile. That makes it the best current candidate for scaling to larger case sets.

One caution is runtime variability. Although DeepSeek passed everything, at least one `oob` run showed a very large end-to-end time outlier (`tiledMatmul` around `10158s`). So the model currently looks strongest on correctness, but not necessarily the most predictable on wall-clock time.

### Kimi K3

`Kimi K3` is the second-strongest model overall with `17/20` passes. Its `oob` performance is especially strong at `9/10`, and it is also relatively efficient in token use compared with the other non-DeepSeek models.

Its main weakness is that `with-sycl-skill` does not improve the pilot. In fact, skill conditioning reduces success from `9/10` to `8/10` and increases both mean tokens and mean runtime. This suggests that Kimi already performs well in the plain TRAE workflow and may be getting less benefit from the added skill context than the other models.

### GLM 5.2

`GLM 5.2` finishes third overall with `16/20` passes, but its behavior is more nuanced than the topline rank suggests. In `oob`, it only reached `7/10`, with several `missing` outcomes. Under `with-sycl-skill`, however, it improved to `9/10`, which is the biggest positive skill uplift among the four models.

That makes GLM the strongest example in this pilot of a model that benefits materially from skill conditioning. It is weaker than DeepSeek on raw robustness, but it appears more steerable than Minimax and more skill-responsive than Kimi.

### Minimax M3

`Minimax M3` is currently the weakest of the four in this pilot, with `13/20` passes. It also has the broadest failure profile:

- `missing`
- `wrong_output`
- `compile_error`

This is important because it suggests the problem is not just one narrow bottleneck. Minimax is not merely struggling with code generation completion; it is also producing semantically wrong outputs and compile failures on some pilot cases.

Another notable characteristic is that Minimax often runs close to the maximum step budget. This was already visible in the earlier single-case analysis and still appears in the 10-case pilot. The model can clearly solve some cases, but its interaction pattern inside TRAE looks less efficient and less stable than the other three models.

## Skill effect by model

The impact of `with-sycl-skill` is not uniform.

| Model | OOB pass rate | With-skill pass rate | Net effect |
|---|---:|---:|---|
| DeepSeek V4 Pro | 100% | 100% | Neutral on correctness |
| Kimi K3 | 90% | 80% | Negative |
| GLM 5.2 | 70% | 90% | Strong positive |
| Minimax M3 | 70% | 60% | Negative |

This is one of the most important findings from the pilot.

- For `GLM 5.2`, the skill is clearly helpful.
- For `DeepSeek V4 Pro`, the skill is not necessary for correctness, though it may still affect style or runtime.
- For `Kimi K3` and `Minimax M3`, the skill currently appears to hurt more than it helps on this pilot set.

That means TRAE prompt/skill policy should probably not be treated as a universal default. Model-specific tuning is likely necessary.

## Efficiency observations

### Tokens

By rough pilot-level totals:

- `Kimi K3` is the most token-efficient among the non-DeepSeek models in `oob`
- `GLM 5.2` sits in the middle
- `Minimax M3` is relatively expensive for its success rate
- `DeepSeek V4 Pro` uses more tokens than Kimi, but the correctness payoff is much better

More importantly, the direction of the skill effect differs by model:

- `GLM 5.2`: skill improves pass rate without increasing mean tokens dramatically
- `Kimi K3`: skill increases tokens and decreases pass rate
- `Minimax M3`: skill increases tokens and decreases pass rate
- `DeepSeek V4 Pro`: skill increases tokens but does not improve correctness, because correctness is already saturated at `100%`

### Wall-clock time

Wall-clock time is more volatile than pass rate, so it should be interpreted carefully. Some runs show obvious outliers, and those likely reflect workflow dynamics, remote execution variance, or long repeated tool loops rather than pure model quality.

Still, a few patterns are visible:

- `DeepSeek V4 Pro` is robust but can occasionally be very slow on individual cases
- `Kimi K3` is reasonably balanced in `oob`, but slower in `with-sycl-skill`
- `GLM 5.2` becomes much more usable under `with-sycl-skill`
- `Minimax M3` remains the least attractive trade-off in this pilot, combining weaker pass rate with relatively heavy interaction behavior

## Failure pattern interpretation

The failure shapes differ meaningfully across models.

### DeepSeek V4 Pro

No scored failures in this pilot. This is the cleanest profile.

### Kimi K3

Only `missing` failures remain. This is a relatively narrow failure mode and is easier to diagnose than a mixed profile involving compile and semantic correctness issues.

### GLM 5.2

Also mostly `missing`, especially in `oob`. This suggests the model sometimes fails to complete a usable migration, but when it does complete, it is often correct. That is consistent with the observation that skill conditioning helps GLM a lot.

### Minimax M3

The most concerning profile:

- `missing`
- `wrong_output`
- `compile_error`

That implies the current weakness is more fundamental than simple incompletion. Minimax may still be useful for specific easy or medium cases, but it is the least reliable overall in the current TRAE setup.

## Interpretation of harness errors

The transient `harness_error` cells seen during pilot execution should not be treated as model failures. Log inspection showed that these were caused by Windows-side file-lock issues such as:

- `WinError 32`
- `The process cannot access the file because it is being used by another process`

These errors occurred after the model-side run had already completed and were associated with artifact collection or local file handling, not with the model’s semantic behavior. After reruns, the affected cells converted to normal scored results. For reporting purposes, they are best categorized as infrastructure-side transient errors rather than model weaknesses.

## Practical recommendations

### Best current model

If the goal is to choose one model to scale first inside TRAE, the answer is:

**DeepSeek V4 Pro**

It has the best correctness, the cleanest failure profile, and the strongest pilot-level robustness.

### Best skill-sensitive model

If the goal is to study whether `with-sycl-skill` adds value, the best model to watch is:

**GLM 5.2**

It is the clearest example in this pilot where the skill improves outcomes materially.

### Best lightweight baseline after DeepSeek

If the goal is to keep a second model that is fairly strong without being the top model:

**Kimi K3**

It has strong `oob` performance and a simpler failure profile than Minimax.

### Least promising current option

Based on this pilot alone:

**Minimax M3**

It is currently the weakest trade-off in the TRAE framework, both in correctness and in failure diversity.

## Bottom line

The 10-case TRAE pilot establishes a clear early picture:

1. `DeepSeek V4 Pro` is the strongest model in this framework.
2. `Kimi K3` is the second-best overall, especially in `oob`.
3. `GLM 5.2` is weaker overall than Kimi, but it benefits substantially from `with-sycl-skill`.
4. `Minimax M3` is currently the weakest of the four in both success rate and stability.
5. Skill conditioning is model-dependent and should not be treated as universally beneficial.
6. Residual `harness_error` issues are infrastructure-side and should be handled separately from model-quality analysis.

This pilot is large enough to support preliminary ranking and workflow conclusions, but final claims should still wait for a larger case set beyond the current 10-case subset.
