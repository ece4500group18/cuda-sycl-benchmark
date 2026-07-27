# Stage 2 migration report: trae-minimax-m3-pilot-v1

Generated: 2026-07-27T09:02:46.241427+00:00

- Results: 2 (2 scored, 0 synthetic)
- Scored migrations passed: 0
- Overall pass rate: 0.000

- Total measured tokens: 1276756
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | minimax-m3 | oob | 1 | 0 | 0.000 | 551.430 | 551.430 | 495208.0 | n/a | unavailable |
| trae | minimax-m3 | with-sycl-skill | 1 | 0 | 0.000 | 437.174 | 437.174 | 781548.0 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| compile_error | 1 |
| wrong_output | 1 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
