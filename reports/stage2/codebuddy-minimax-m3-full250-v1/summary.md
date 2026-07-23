# Stage 2 migration report: codebuddy-minimax-m3-full250-v1

Generated: 2026-07-21T08:00:35.606353+00:00

- Results: 107 (107 scored, 0 synthetic)
- Scored migrations passed: 97
- Overall pass rate: 0.907

- Total measured tokens: 82458541
- Cost USD by source: provider_reported=0.000000

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| codebuddy | minimax-m3 | oob | 54 | 48 | 0.889 | 100.457 | 83.225 | 692414.9 | 0.000000 | provider_reported |
| codebuddy | minimax-m3 | with-sycl-skill | 53 | 49 | 0.925 | 94.930 | 88.947 | 850342.2 | 0.000000 | provider_reported |

## Failure funnel

| status | count |
| --- | ---: |
| compile_error | 1 |
| pass | 97 |
| wrong_output | 9 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
