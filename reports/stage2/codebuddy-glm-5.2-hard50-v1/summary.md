# Stage 2 migration report: codebuddy-glm-5.2-hard50-v1

Generated: 2026-07-23T02:25:42.432170+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 90
- Overall pass rate: 0.900

- Total measured tokens: 93731050
- Cost USD by source: provider_reported=0.000000

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| codebuddy | glm-5.2 | oob | 50 | 45 | 0.900 | 177.471 | 160.161 | 740093.5 | 0.000000 | provider_reported |
| codebuddy | glm-5.2 | with-sycl-skill | 50 | 45 | 0.900 | 213.350 | 205.278 | 1134527.5 | 0.000000 | provider_reported |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 90 |
| wrong_output | 10 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
