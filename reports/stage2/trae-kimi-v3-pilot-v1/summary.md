# Stage 2 migration report: trae-kimi-v3-pilot-v1

Generated: 2026-07-28T19:54:51.482357+00:00

- Results: 20 (20 scored, 0 synthetic)
- Scored migrations passed: 17
- Overall pass rate: 0.850

- Total measured tokens: 4577417
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | kimi-k3 | oob | 10 | 9 | 0.900 | 523.944 | 476.751 | 188400.2 | n/a | unavailable |
| trae | kimi-k3 | with-sycl-skill | 10 | 8 | 0.800 | 615.294 | 538.482 | 269341.5 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| missing | 3 |
| pass | 17 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
