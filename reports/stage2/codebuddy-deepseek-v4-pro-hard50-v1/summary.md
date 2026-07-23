# Stage 2 migration report: codebuddy-deepseek-v4-pro-hard50-v1

Generated: 2026-07-23T02:22:04.696386+00:00

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 88
- Overall pass rate: 0.880

- Total measured tokens: 86023233
- Cost USD by source: provider_reported=0.000000

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| codebuddy | deepseek-v4-pro | oob | 50 | 43 | 0.860 | 157.913 | 145.754 | 802198.3 | 0.000000 | provider_reported |
| codebuddy | deepseek-v4-pro | with-sycl-skill | 50 | 45 | 0.900 | 169.157 | 156.965 | 918266.3 | 0.000000 | provider_reported |

## Failure funnel

| status | count |
| --- | ---: |
| compile_error | 1 |
| pass | 88 |
| wrong_output | 11 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
