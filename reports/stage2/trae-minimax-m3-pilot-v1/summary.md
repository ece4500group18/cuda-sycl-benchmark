# Stage 2 migration report: trae-minimax-m3-pilot-v1

Generated: 2026-07-28T02:39:35.814873+00:00

- Results: 20 (20 scored, 0 synthetic)
- Scored migrations passed: 13
- Overall pass rate: 0.650

- Total measured tokens: 6533180
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| trae | minimax-m3 | oob | 10 | 7 | 0.700 | 428.147 | 321.529 | 313068.4 | n/a | unavailable |
| trae | minimax-m3 | with-sycl-skill | 10 | 6 | 0.600 | 3803.238 | 347.451 | 340249.6 | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| compile_error | 2 |
| missing | 4 |
| pass | 13 |
| wrong_output | 1 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
