# Stage 2 migration report: baseline-claude-opus47-local-v1

Generated: 2026-08-03T03:51:32.097363+00:00

- Results: 118 (118 scored, 0 synthetic)
- Scored migrations passed: 104
- Overall pass rate: 0.881

- Total measured tokens: 233029067
- Cost USD by source: provider_reported=166.273760

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| claude-code | opus-4-7 | oob | 59 | 52 | 0.881 | 207.512 | 208.465 | 1967410.5 | 82.687825 | provider_reported |
| claude-code | opus-4-7 | with-sycl-skill | 59 | 52 | 0.881 | 203.282 | 192.776 | 1982234.7 | 83.585935 | provider_reported |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 104 |
| run_error | 4 |
| wrong_output | 10 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
