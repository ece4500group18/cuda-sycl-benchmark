# Stage 2 migration report: trae-kimi-v3-pilot-v1

Generated: 2026-07-28T17:26:24.716051+00:00

- Results: 18 (18 scored, 0 synthetic)
- Scored migrations passed: 15
- Overall pass rate: 0.833

- Total measured tokens: 4105406
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | kimi-k3 | oob | 8 | 7 | 0.875 | 482.662 | 452.211 | 176498.9 | n/a | unavailable |
| trae | kimi-k3 | with-sycl-skill | 10 | 8 | 0.800 | 615.294 | 538.482 | 269341.5 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| missing | 3 |
| pass | 15 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
