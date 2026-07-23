# Stage 2 migration report: codebuddy-minimax-m3-hard50-v1

Generated: 2026-07-21T10:38:15.904298+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 89
- Overall pass rate: 0.890

- Total measured tokens: 72737692
- Cost USD by source: provider_reported=0.000000

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| codebuddy | minimax-m3 | oob | 50 | 45 | 0.900 | 86.927 | 82.244 | 607102.1 | 0.000000 | provider_reported |
| codebuddy | minimax-m3 | with-sycl-skill | 50 | 44 | 0.880 | 98.913 | 90.105 | 847651.8 | 0.000000 | provider_reported |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 89 |
| wrong_output | 11 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
