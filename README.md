# SYCL Bridge

Key words: CUDA, SYCL, coding agent, migration, benchmarking

## Hard-50 Test Results

All rates are computed per condition over **50 hard cases** (50 scored cells). Each row corresponds to one `harness × model × condition` slice.

| harness | model | condition | attempts | mean_tokens/cell | pass | missing | compile_error | wrong_output | run_error | harness_error | other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude_code | claude-opus-4-7 | oob | 50 | n/a | 88.0% (44/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 2.0% (1/50) | 0.0% (0/50) | 0.0% (0/50) |
| claude_code | claude-opus-4-7 | with-sycl-skill | 50 | n/a | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | deepseek-v4-pro | oob | 50 | 802,198 | 86.0% (43/50) | 0.0% (0/50) | 2.0% (1/50) | 12.0% (6/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | deepseek-v4-pro | with-sycl-skill | 50 | 918,266 | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | glm-5.2 | oob | 50 | 740,094 | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | glm-5.2 | with-sycl-skill | 50 | 1,134,527 | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | kimi-k3-2 | oob | 50 | 537,634 | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | kimi-k3-2 | with-sycl-skill | 50 | 520,326 | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | minimax-m3 | oob | 50 | 607,102 | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| codebuddy | minimax-m3 | with-sycl-skill | 50 | 847,652 | 88.0% (44/50) | 0.0% (0/50) | 0.0% (0/50) | 12.0% (6/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | deepseek-v4pro | oob | 50 | n/a | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | deepseek-v4pro | with-sycl-skill | 50 | n/a | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | glm-52 | oob | 50 | n/a | 94.0% (47/50) | 0.0% (0/50) | 0.0% (0/50) | 6.0% (3/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | glm-52 | with-sycl-skill | 50 | n/a | 92.0% (46/50) | 0.0% (0/50) | 0.0% (0/50) | 8.0% (4/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | kimi-k3 | oob | 50 | n/a | 98.0% (49/50) | 0.0% (0/50) | 0.0% (0/50) | 2.0% (1/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | kimi-k3 | with-sycl-skill | 50 | n/a | 94.0% (47/50) | 0.0% (0/50) | 0.0% (0/50) | 6.0% (3/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | minimax-m3 | oob | 50 | n/a | 90.0% (45/50) | 0.0% (0/50) | 0.0% (0/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| qoder | minimax-m3 | with-sycl-skill | 50 | n/a | 88.0% (44/50) | 0.0% (0/50) | 0.0% (0/50) | 12.0% (6/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| trae | deepseek-v4pro | oob | 50 | 394,031 | 60.0% (30/50) | 8.0% (4/50) | 22.0% (11/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| trae | deepseek-v4pro | with-sycl-skill | 50 | 372,925 | 60.0% (30/50) | 14.0% (7/50) | 16.0% (8/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| trae | glm-5.2 | oob | 50 | 203,048 | 60.0% (30/50) | 28.0% (14/50) | 2.0% (1/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |
| trae | glm-5.2 | with-sycl-skill | 50 | 303,004 | 60.0% (30/50) | 22.0% (11/50) | 8.0% (4/50) | 10.0% (5/50) | 0.0% (0/50) | 0.0% (0/50) | 0.0% (0/50) |

## Sources

- `reports/stage2/trae-deepseek-v4pro-hard50-v1/summary.{md,json,csv}` (this branch)
- `reports/stage2/trae-glm-5.2-hard50-v1/summary.{md,json,csv}` (this branch)
- `origin/yuepan:reports/stage2/codebuddy-deepseek-v4-pro-hard50-v1/summary.{md,json,csv}`
- `origin/yuepan:reports/stage2/codebuddy-glm-5.2-hard50-v1/summary.{md,json,csv}`
- `origin/yuepan:reports/stage2/codebuddy-kimi-k3-2-hard50-v1/summary.{md,json,csv}`
- `origin/yuepan:reports/stage2/codebuddy-minimax-m3-hard50-v1/summary.{md,json,csv}`
- `origin/Weixuan_stage2:reports/stage2/qoder-deepseek-v4pro-hard50-v1/summary.{md,json,csv}`
- `origin/Weixuan_stage2:reports/stage2/qoder-glm-52-hard50-v1/summary.{md,json,csv}`
- `origin/Weixuan_stage2:reports/stage2/qoder-kimi-k3-hard50-v1/summary.{md,json,csv}`
- `origin/Weixuan_stage2:reports/stage2/qoder-minimax-m3-hard50-v1/summary.{md,json,csv}`
- `origin/stage2-hard50-results:reports/stage2/baseline-claude-opus47-local-v1/RESULTS.md` (hard-50 only; `summary.json` mixes pilot + hard-50)
