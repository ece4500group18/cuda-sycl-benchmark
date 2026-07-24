# Stage 2 migration report: qoder-minimax-m3-hard50-v1

Generated: 2026-07-22T20:53:43.757425+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 88
- Overall pass rate: 0.880

- Total measured tokens: 0
- Cost USD by source: n/a

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | minimax-m3 | oob | 50 | 44 | 0.880 | 175.196 | 90.447 | n/a | n/a | unavailable |
| qoder | minimax-m3 | with-sycl-skill | 50 | 44 | 0.880 | 114.718 | 96.965 | n/a | n/a | unavailable |

## Failure funnel

| status | count |
| --- | ---: |
| missing | 1 |
| pass | 88 |
| wrong_output | 11 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
