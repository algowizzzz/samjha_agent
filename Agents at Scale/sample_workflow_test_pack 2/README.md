Sample Workflow Test Pack
========================

This pack is designed to test:
- Upload types: TXT, MD, PDF, DOCX, CSV
- Programmatic ingestion (ignore images, keep tables)
- Vision ingestion (requires vision_ingestion_prompt.md)
- Prompt chain steps (R0 -> R1 -> R2 -> R3)
- CSV row-per-task pipeline + compiled CSV output

Files
-----
Inputs:
- input_vendor_risk_policy_excerpt.txt
- input_incident_response_sop_excerpt.md
- input_policy_with_table.pdf
- input_data_retention_standard_excerpt.docx
- input_tasks.csv  (each row = task)

Prompts (upload as .md in Admin):
- prompts/step_01_extract_obligations.md
- prompts/step_02_gap_analysis.md
- prompts/step_03_markdown_report.md
- prompts/csv_row_task_processor.md
- prompts/vision_ingestion_prompt.md

Suggested test runs
-------------------
A) Programmatic ingestion + Markdown export
   - Workflow: "Policy Obligation Extractor"
   - Input: PDF/DOCX/TXT/MD
   - Chain: Step 1 -> Step 2 -> Step 3
   - Export: MD

B) Vision ingestion + Markdown export
   - Workflow: "Vision OCR to Markdown"
   - Input: PDF
   - Ingestion mode: Vision
   - Vision prompt: prompts/vision_ingestion_prompt.md
   - Export: MD (programmatic final conversion if needed)

C) CSV pipeline + JSON per row + compiled CSV export
   - Workflow: "CSV Ticket Router"
   - Input: input_tasks.csv
   - Chain: prompts/csv_row_task_processor.md
   - Export: CSV (compiled from per-row JSON outputs)

Notes
-----
- These are intentionally small and safe for quick end-to-end testing.

Golden Outputs
-------------
- expected_outputs/ contains sample "golden" R0/R1/R2/R3 artifacts and compiled CSV export for regression testing.
