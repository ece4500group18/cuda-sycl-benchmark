# Stage 2 migration report: trae-glm-5.2-hard50-v1

Generated: 2026-08-01T10:10:56.952816+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 60
- Overall pass rate: 0.600

- Total measured tokens: 25302597
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | glm-5.2 | oob | 50 | 30 | 0.600 | 335.189 | 308.979 | 203047.5 | n/a | unavailable |
| trae | glm-5.2 | with-sycl-skill | 50 | 30 | 0.600 | 815.946 | 279.407 | 303004.4 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| compile_error | 5 |
| missing | 25 |
| pass | 60 |
| wrong_output | 10 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
