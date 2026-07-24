# Stage 2 migration report: qoder-kimi-k3-hard50-v1

Generated: 2026-07-24T08:38:57.444870+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 96
- Overall pass rate: 0.960

- Total measured tokens: 0
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | kimi-k3 | oob | 50 | 49 | 0.980 | 376.271 | 207.615 | n/a | n/a | unavailable |
| qoder | kimi-k3 | with-sycl-skill | 50 | 47 | 0.940 | 331.717 | 205.692 | n/a | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 96 |
| wrong_output | 4 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
