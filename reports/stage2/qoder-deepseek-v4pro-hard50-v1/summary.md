# Stage 2 migration report: qoder-deepseek-v4pro-hard50-v1

Generated: 2026-07-23T09:09:47.891115+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 90
- Overall pass rate: 0.900

- Total measured tokens: 0
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | deepseek-v4pro | oob | 50 | 45 | 0.900 | 137.169 | 127.819 | n/a | n/a | unavailable |
| qoder | deepseek-v4pro | with-sycl-skill | 50 | 45 | 0.900 | 138.462 | 121.877 | n/a | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 90 |
| wrong_output | 10 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
