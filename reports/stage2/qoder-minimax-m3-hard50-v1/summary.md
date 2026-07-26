# Stage 2 migration report: qoder-minimax-m3-hard50-v1

Generated: 2026-07-22T20:53:43.757425+00:00
Updated: 2026-07-26 (added credits & cost data from Qoder platform)

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 88
- Overall pass rate: 0.880

- Total measured credits: 282.27
- Cost USD by source: 5.14

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean credits | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | minimax-m3 | oob | 50 | 44 | 0.880 | 175.196 | 90.447 | 2.54 | $2.32 | qoder |
| qoder | minimax-m3 | with-sycl-skill | 50 | 44 | 0.880 | 114.718 | 96.965 | 3.10 | $2.82 | qoder |

## Failure funnel

| status | count |
| --- | ---: |
| missing | 1 |
| pass | 88 |
| wrong_output | 11 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
