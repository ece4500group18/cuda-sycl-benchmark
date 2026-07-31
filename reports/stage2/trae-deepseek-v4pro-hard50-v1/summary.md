# Stage 2 migration report: trae-deepseek-v4pro-hard50-v1

Generated: 2026-07-31T12:36:09.560793+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 57
- Overall pass rate: 0.570

- Total measured tokens: 38705749
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | deepseek-v4pro | oob | 50 | 29 | 0.580 | 1717.591 | 463.263 | 344557.5 | n/a | unavailable |
| trae | deepseek-v4pro | with-sycl-skill | 50 | 28 | 0.560 | 1141.478 | 521.682 | 429557.5 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| compile_error | 13 |
| missing | 21 |
| pass | 57 |
| wrong_output | 9 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
