# Stage 2 migration report: trae-deepseek-v4pro-hard50-v1

Generated: 2026-08-01T09:08:38.659553+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 60
- Overall pass rate: 0.600

- Total measured tokens: 38347823
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | deepseek-v4pro | oob | 50 | 30 | 0.600 | 1748.929 | 453.429 | 394031.5 | n/a | unavailable |
| trae | deepseek-v4pro | with-sycl-skill | 50 | 30 | 0.600 | 597.322 | 479.808 | 372925.0 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| compile_error | 19 |
| missing | 11 |
| pass | 60 |
| wrong_output | 10 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
