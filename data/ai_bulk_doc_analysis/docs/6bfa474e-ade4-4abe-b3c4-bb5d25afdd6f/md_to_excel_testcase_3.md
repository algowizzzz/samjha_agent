# Test Case 3 — Edge-ish Cases (Still Valid Markdown)

This file includes:
- extra spaces around pipes
- empty cells
- long text cells
All tables are still standard markdown tables.

## Table X — Survey Results (Empty Cells)

| Respondent | Score | Comment |
|---|---:|---|
| R-001 | 5 | Loved it |
| R-002 | 3 |  |
| R-003 | 4 | Pretty good overall |

Between tables: plain text paragraph describing what comes next.

## Table Y — Inventory (Extra Spaces)

| Item ID | Item Name | Qty | Warehouse |
| --- | --- | ---: | --- |
| I-10 | Widget Alpha | 120 | WH-A |
| I-11 | Widget Beta | 0 | WH-B |
| I-12 | Widget Gamma | 15 | WH-A |

## Table Z — Long Text Cells

| Ticket | Owner | Description |
|---|---|---|
| INC-101 | Saad | User reports export issues when tables contain many rows; this is a long cell meant to test wrapping. |
| INC-102 | Team Ops | Scheduled maintenance window; confirm that the converter keeps full cell text intact. |
| INC-103 | QA | Verify that each table becomes its own sheet and that headers are preserved. |
