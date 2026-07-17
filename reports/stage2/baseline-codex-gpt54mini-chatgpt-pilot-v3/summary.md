# Stage 2 migration report: baseline-codex-gpt54mini-chatgpt-pilot-v3

Generated: 2026-07-17T11:50:17.276691+00:00

- Results: 2 (2 scored, 0 synthetic)
- Scored migrations passed: 2
- Overall pass rate: 1.000

- Total measured tokens: 305633
- Cost USD by source: api_price_estimate=0.078600

## Harness x model x skill KPIs

| harness | model | condition | scored | passed | pass rate | mean E2E s | median E2E s | mean tokens | total cost USD | cost source |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| codex-cli | gpt-5.4-mini | oob | 1 | 1 | 1.000 | 229.538 | 229.538 | 154413.0 | 0.034568 | api_price_estimate |
| codex-cli | gpt-5.4-mini | with-sycl-skill | 1 | 1 | 1.000 | 258.780 | 258.780 | 151220.0 | 0.044032 | api_price_estimate |

## Failure funnel

| status | count |
| --- | ---: |
| pass | 2 |

> Synthetic mock results test orchestration only and are excluded from all scored KPIs.
