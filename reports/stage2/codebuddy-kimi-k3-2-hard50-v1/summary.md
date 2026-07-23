# Stage 2 migration report: codebuddy-kimi-k3-2-hard50-v1

Generated: 2026-07-23T02:45:33.614704+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 90
- Overall pass rate: 0.900

- Total measured tokens: 52897988
- Cost USD by source: provider_reported=0.000000

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| codebuddy | kimi-k3-2 | oob | 50 | 45 | 0.900 | 219.054 | 182.717 | 537633.6 | 0.000000 | provider_reported |
| codebuddy | kimi-k3-2 | with-sycl-skill | 50 | 45 | 0.900 | 202.722 | 157.285 | 520326.2 | 0.000000 | provider_reported |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 90 |
| wrong_output | 10 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
