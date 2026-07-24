# Stage 2 migration report: qoder-glm-52-hard50-v1

Generated: 2026-07-23T04:48:28.682422+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 93
- Overall pass rate: 0.930

- Total measured tokens: 0
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | glm-52 | oob | 50 | 47 | 0.940 | 270.837 | 173.309 | n/a | n/a | unavailable |
| qoder | glm-52 | with-sycl-skill | 50 | 46 | 0.920 | 262.490 | 170.302 | n/a | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 93 |
| wrong_output | 7 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
