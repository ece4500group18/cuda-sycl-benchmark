# Stage 2 migration report: qoder-kimi-k3-hard50-v1

Generated: 2026-07-24T08:38:57.444870+00:00
Updated: 2026-07-26 (added credits & cost data from Qoder platform; includes re-evaluated results)

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 96
- Overall pass rate: 0.960

- Total measured credits: 1817.92
- Cost USD by source: 23.60

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean credits | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | kimi-k3 | oob | 50 | 49 | 0.980 | 376.271 | 207.615 | 19.04 | $12.36 | qoder |
| qoder | kimi-k3 | with-sycl-skill | 50 | 47 | 0.940 | 331.717 | 205.692 | 17.32 | $11.24 | qoder |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 96 |
| wrong_output | 4 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
