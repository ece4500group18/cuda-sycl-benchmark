# Stage 2 migration report: trae-glm-5.2-pilot-v1

Generated: 2026-07-28T19:31:36.066759+00:00

- Results: 20 (20 scored, 0 synthetic)
- Scored migrations passed: 16
- Overall pass rate: 0.800

- Total measured tokens: 5641677
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | glm-5.2 | oob | 10 | 7 | 0.700 | 937.518 | 824.482 | 287778.2 | n/a | unavailable |
| trae | glm-5.2 | with-sycl-skill | 10 | 9 | 0.900 | 309.350 | 289.599 | 276389.5 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| missing | 4 |
| pass | 16 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
