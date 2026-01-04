# Test Case 2 — Mixed Formats and Longer Text

This file has multiple sections and **four** tables.
It also includes commas, decimals, and percentages for parsing.

## Table 1 — KPI Dashboard

| KPI | Value | Unit | As Of |
|---|---:|---|---|
| Conversion Rate | 2.75 | % | 2025-12-31 |
| Avg Order Value | 48.90 | USD | 2025-12-31 |
| Net Promoter Score | 41 | points | 2025-12-31 |
| Churn | 0.8 | % | 2025-12-31 |

Below is some explanatory text. The extractor should only pick up table rows containing the `|` character.

## Table 2 — Budget (With Currency Symbols)

| Cost Center | Item | Amount |
|---|---|---:|
| Marketing | Ads | 12500 |
| Cloud | Hosting | 3200 |
| Payroll | Contractors | 18000 |
| Legal | Review | 2100 |

## Table 3 — Feature Flags

| Feature | Enabled | Rollout_Percent |
|---|---|---:|
| new_homepage | TRUE | 100 |
| beta_search | TRUE | 25 |
| md_to_excel | FALSE | 0 |

Some more text here to ensure non-table content is handled.

## Table 4 — Dates and Status

| ID | Start_Date | End_Date | Status |
|---|---|---|---|
| T-001 | 2025-11-01 | 2025-11-15 | Done |
| T-002 | 2025-11-16 | 2025-12-05 | Done |
| T-003 | 2025-12-06 | 2026-01-15 | In Progress |
