# Stage 2 migration report: qoder-deepseek-v4pro-hard50-v1

Generated: 2026-07-23T09:09:33.891456+00:00
Updated: 2026-07-24 (added credits & cost data from Qoder platform)

- Results: 100 (100 scored, 0 synthetic)
- Scored migrations passed: 90
- Overall pass rate: 0.900

- Total measured credits: 689.18
- Cost USD by source: 6.38

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean credits | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| qoder | deepseek-v4pro | oob | 50 | 45 | 0.900 | 148.746 | 107.786 | 6.89 | $3.19 | qoder |
| qoder | deepseek-v4pro | with-sycl-skill | 50 | 45 | 0.900 | 158.954 | 107.847 | 6.89 | $3.19 | qoder |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 90 |
| wrong_output | 10 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
