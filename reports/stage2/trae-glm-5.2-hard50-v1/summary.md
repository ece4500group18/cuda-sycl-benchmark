# Stage 2 migration report: trae-glm-5.2-hard50-v1

Generated: 2026-07-31T08:50:02.345678+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 58
- Overall pass rate: 0.580

- Total measured tokens: 24107625
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | glm-5.2 | oob | 50 | 30 | 0.600 | 440.742 | 331.303 | 220747.7 | n/a | unavailable |
| trae | glm-5.2 | with-sycl-skill | 50 | 28 | 0.560 | 970.880 | 279.407 | 261404.8 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| missing | 33 |
| pass | 58 |
| wrong_output | 9 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
