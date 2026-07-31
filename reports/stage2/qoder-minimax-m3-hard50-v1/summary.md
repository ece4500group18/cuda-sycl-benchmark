# Stage 2 migration report: qoder-minimax-m3-hard50-v1

Generated: 2026-07-31T14:10:00.047330+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 89
- Overall pass rate: 0.890

- Total measured tokens: 282.27
- Cost USD by source: 5.14

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | minimax-m3 | oob | 50 | 45 | 0.900 | 175.196 | 90.447 | 2.54 | $2.32 | qoder |
| qoder | minimax-m3 | with-sycl-skill | 50 | 44 | 0.880 | 114.718 | 96.965 | 3.10 | $2.82 | qoder |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 89 |
| wrong_output | 11 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
