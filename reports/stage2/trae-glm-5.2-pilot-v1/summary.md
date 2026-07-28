# Stage 2 migration report: trae-glm-5.2-pilot-v1

Generated: 2026-07-28T13:52:54.691619+00:00

- Results: 17 (17 scored, 0 synthetic)
- Scored migrations passed: 13
- Overall pass rate: 0.765

- Total measured tokens: 4805715
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | glm-5.2 | oob | 10 | 7 | 0.700 | 937.518 | 824.482 | 287778.2 | n/a | unavailable |
| trae | glm-5.2 | with-sycl-skill | 7 | 6 | 0.857 | 339.569 | 313.141 | 275419.0 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| missing | 4 |
| pass | 13 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
