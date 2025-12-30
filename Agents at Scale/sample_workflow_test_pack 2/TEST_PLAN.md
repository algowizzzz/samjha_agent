# Bulk Document Analysis - Test Plan

Based on the sample workflow test pack, here are all available tests organized by category.

## Test Categories

### A. Programmatic Ingestion Tests

#### A1. PDF Programmatic Ingestion + Single-Step Chain
- **Input**: `input_policy_with_table.pdf`
- **Ingestion**: Programmatic (PDF → Markdown)
- **Chain**: 1 step - Extract Obligations (`step_01_extract_obligations.md`)
- **Export**: Markdown
- **Expected**: JSON with obligations array
- **Status**: ✅ **COMPLETED** (tested successfully)

#### A2. DOCX Programmatic Ingestion + Single-Step Chain
- **Input**: `input_data_retention_standard_excerpt.docx`
- **Ingestion**: Programmatic (DOCX → Markdown)
- **Chain**: 1 step - Extract Obligations
- **Export**: Markdown
- **Expected**: JSON with obligations extracted from DOCX

#### A3. TXT Programmatic Ingestion + Single-Step Chain
- **Input**: `input_vendor_risk_policy_excerpt.txt`
- **Ingestion**: Programmatic (TXT → Markdown)
- **Chain**: 1 step - Extract Obligations
- **Export**: Markdown
- **Expected**: JSON with obligations extracted from plain text

#### A4. MD Programmatic Ingestion + Single-Step Chain
- **Input**: `input_incident_response_sop_excerpt.md`
- **Ingestion**: Programmatic (MD → Markdown, minimal processing)
- **Chain**: 1 step - Extract Obligations
- **Export**: Markdown
- **Expected**: JSON with obligations extracted from Markdown

### B. Multi-Step Chain Tests

#### B1. 3-Step Obligation Analysis Chain (PDF)
- **Input**: `input_policy_with_table.pdf`
- **Ingestion**: Programmatic
- **Chain**: 
  - Step 1: Extract Obligations (`step_01_extract_obligations.md`)
  - Step 2: Gap Analysis (`step_02_gap_analysis.md`)
  - Step 3: Markdown Report (`step_03_markdown_report.md`)
- **Export**: Markdown
- **Expected**: 
  - R1: JSON obligations
  - R2: JSON gap analysis
  - R3: Final markdown report
- **Golden Output**: `expected_outputs/policy_with_table_R1_obligations.json`, `R2_gaps.json`, `R3_report.md`

#### B2. 3-Step Chain with Different Input Types
- **Input**: DOCX, TXT, or MD files
- **Ingestion**: Programmatic
- **Chain**: Same 3-step chain as B1
- **Export**: Markdown
- **Expected**: Complete analysis pipeline for each file type

### C. Vision Ingestion Tests

#### C1. PDF Vision Ingestion (Images & Visuals)
- **Input**: `input_policy_with_table.pdf` (contains tables/images)
- **Ingestion**: Vision mode (uses `vision_ingestion_prompt.md`)
- **Chain**: 1 step - Extract Obligations
- **Export**: Markdown
- **Expected**: Better extraction of visual elements, tables, diagrams
- **Note**: Requires vision prompt stored in DB

#### C2. Vision Ingestion + Multi-Step Chain
- **Input**: PDF with complex visuals
- **Ingestion**: Vision mode
- **Chain**: 3-step analysis chain
- **Export**: Markdown
- **Expected**: Complete analysis with visual content preserved

### D. CSV Row-Per-Task Pipeline Tests

#### D1. CSV Ingestion + Per-Row Processing
- **Input**: `input_tasks.csv`
- **Ingestion**: CSV mode (each row = independent task)
- **Chain**: 1 step - CSV Row Task Processor (`csv_row_task_processor.md`)
- **Export**: CSV (compiled from per-row JSON outputs)
- **Expected**: 
  - Each CSV row processed independently
  - Per-row JSON outputs (e.g., `TCK-001.json`, `TCK-002.json`)
  - Compiled CSV export (`compiled_output.csv`)
- **Golden Output**: `expected_outputs/csv_ticket_router/`

#### D2. CSV + Multi-Step Chain
- **Input**: `input_tasks.csv`
- **Ingestion**: CSV mode
- **Chain**: Multi-step chain (e.g., 3-step analysis)
- **Export**: CSV
- **Expected**: Each row goes through all chain steps, final CSV compiled

### E. Export Format Tests

#### E1. Markdown Export
- **Input**: Any document type
- **Chain**: Any chain
- **Export**: MD format
- **Expected**: Markdown file with results
- **Golden Output**: `expected_outputs/export_examples/policy_with_table_export.md`

#### E2. JSON Export
- **Input**: Any document type
- **Chain**: Any chain (preferably JSON-output steps)
- **Export**: JSON format
- **Expected**: JSON file with structured results

#### E3. CSV Export
- **Input**: CSV or any document type
- **Chain**: Any chain
- **Export**: CSV format
- **Expected**: CSV file with tabular results
- **Golden Output**: `expected_outputs/export_examples/csv_ticket_router_export.csv`

#### E4. DOCX Export
- **Input**: Any document type
- **Chain**: Any chain
- **Export**: DOCX format
- **Expected**: Word document with formatted results

#### E5. PDF Export
- **Input**: Any document type
- **Chain**: Any chain
- **Export**: PDF format
- **Expected**: PDF document with formatted results

### F. Workflow System Tests

#### F1. Workflow Creation via UI
- **Test**: Create workflow using Admin UI
- **Steps**: 
  1. Create ingestion profile
  2. Create export profile
  3. Create chain
  4. Create workflow combining all
- **Expected**: Workflow saved and accessible

#### F2. Workflow Execution via Runner UI
- **Test**: Execute workflow from Runner UI
- **Steps**:
  1. Select workflow
  2. Upload documents
  3. Run workflow
  4. Monitor progress
  5. Download results
- **Expected**: Complete workflow execution

#### F3. Workflow Versioning
- **Test**: Update workflow creates new version
- **Steps**:
  1. Create workflow v1
  2. Update workflow (change chain/ingestion/export)
  3. Verify v2 created, v1 preserved
- **Expected**: Immutable versioning works

### G. Domain & Access Control Tests

#### G1. Domain-Based Workflow Visibility
- **Test**: Workflows visible only to users in same domain
- **Steps**:
  1. Create workflow with domains ["Risk", "Compliance"]
  2. Login as user with domain ["Risk"]
  3. Verify workflow visible
  4. Login as user with domain ["Finance"]
  5. Verify workflow not visible
- **Expected**: Domain filtering works

#### G2. Super Admin Access
- **Test**: Super admin sees all workflows
- **Steps**:
  1. Create workflows in different domains
  2. Login as super admin
  3. Verify all workflows visible
- **Expected**: Super admin bypasses domain restrictions

#### G3. Domain Admin Access
- **Test**: Domain admin manages workflows in their domain
- **Steps**:
  1. Create workflow in domain ["Risk"]
  2. Login as Risk domain admin
  3. Verify can edit/delete workflow
  4. Login as Compliance domain admin
  5. Verify cannot edit/delete Risk workflow
- **Expected**: Domain admin permissions work

### H. Error Handling & Edge Cases

#### H1. Invalid File Type
- **Test**: Upload file type not in ingestion profile
- **Input**: Upload TXT when profile only accepts PDF
- **Expected**: Error message, file rejected

#### H2. Large File Handling
- **Test**: Process large PDF (if available)
- **Input**: Large PDF file
- **Expected**: Handles gracefully, shows progress

#### H3. Chain Step Failure
- **Test**: Chain step fails (e.g., invalid prompt)
- **Expected**: Error captured, run status = ERROR, error message shown

#### H4. Missing Required Inputs
- **Test**: Chain step requires R1 but only R0 exists
- **Expected**: Error message about missing inputs

#### H5. Empty Document
- **Test**: Upload empty/blank PDF
- **Expected**: Handles gracefully, shows appropriate error

### I. Performance & Concurrency Tests

#### I1. Multiple Documents in One Run
- **Test**: Upload multiple PDFs, process in parallel
- **Input**: 3-5 PDF files
- **Expected**: All documents processed, progress tracked per doc

#### I2. Multiple Runs Concurrent
- **Test**: Start multiple runs simultaneously
- **Expected**: All runs process independently, no conflicts

#### I3. CSV with Many Rows
- **Test**: Process CSV with 10+ rows
- **Input**: Large CSV file
- **Expected**: All rows processed, progress tracked per row

### J. Integration Tests

#### J1. End-to-End: PDF → 3-Step Chain → MD Export
- **Test**: Complete workflow from upload to export
- **Input**: `input_policy_with_table.pdf`
- **Workflow**: 
  - Ingestion: PDF programmatic
  - Chain: 3-step obligation analysis
  - Export: Markdown
- **Expected**: Complete pipeline works, output matches golden files

#### J2. End-to-End: CSV → Per-Row Processing → CSV Export
- **Test**: Complete CSV workflow
- **Input**: `input_tasks.csv`
- **Workflow**:
  - Ingestion: CSV
  - Chain: CSV row processor
  - Export: CSV
- **Expected**: Compiled CSV matches golden output

#### J3. End-to-End: Vision PDF → Analysis → JSON Export
- **Test**: Vision ingestion with analysis
- **Input**: PDF with visuals
- **Workflow**:
  - Ingestion: Vision
  - Chain: Analysis chain
  - Export: JSON
- **Expected**: Visual content preserved in analysis

### K. Regression Tests

#### K1. Compare Against Golden Outputs
- **Test**: Run tests and compare outputs to expected_outputs/
- **Files to Compare**:
  - `policy_with_table_R1_obligations.json`
  - `policy_with_table_R2_gaps.json`
  - `policy_with_table_R3_report.md`
  - `csv_ticket_router/compiled_output.csv`
- **Expected**: Outputs match golden files (allowing for minor LLM variations)

## Test Execution Priority

### Phase 1: Core Functionality (Critical)
1. ✅ A1 - PDF Programmatic + Single-Step (COMPLETED)
2. B1 - 3-Step Chain (PDF)
3. D1 - CSV Row-Per-Task Pipeline
4. J1 - End-to-End PDF → 3-Step → MD

### Phase 2: Input Types (High Priority)
5. A2 - DOCX Ingestion
6. A3 - TXT Ingestion
7. A4 - MD Ingestion
8. C1 - Vision Ingestion

### Phase 3: Export Formats (Medium Priority)
9. E1 - Markdown Export
10. E2 - JSON Export
11. E3 - CSV Export
12. E4 - DOCX Export
13. E5 - PDF Export

### Phase 4: Workflow System (Medium Priority)
14. F1 - Workflow Creation via UI
15. F2 - Workflow Execution via Runner UI
16. F3 - Workflow Versioning

### Phase 5: Access Control (Medium Priority)
17. G1 - Domain-Based Visibility
18. G2 - Super Admin Access
19. G3 - Domain Admin Access

### Phase 6: Error Handling (Low Priority)
20. H1-H5 - Error scenarios

### Phase 7: Performance (Low Priority)
21. I1-I3 - Concurrency and scale

### Phase 8: Regression (Ongoing)
22. K1 - Golden Output Comparison

## Test Data Files

### Input Files
- `input_policy_with_table.pdf` - PDF with tables
- `input_data_retention_standard_excerpt.docx` - DOCX document
- `input_vendor_risk_policy_excerpt.txt` - Plain text
- `input_incident_response_sop_excerpt.md` - Markdown
- `input_tasks.csv` - CSV with multiple rows

### Prompts
- `step_01_extract_obligations.md` - Extract obligations
- `step_02_gap_analysis.md` - Gap analysis
- `step_03_markdown_report.md` - Generate report
- `csv_row_task_processor.md` - Process CSV rows
- `vision_ingestion_prompt.md` - Vision mode prompt

### Expected Outputs (Golden Files)
- `expected_outputs/policy_with_table_R0_normalized.md`
- `expected_outputs/policy_with_table_R1_obligations.json`
- `expected_outputs/policy_with_table_R2_gaps.json`
- `expected_outputs/policy_with_table_R3_report.md`
- `expected_outputs/csv_ticket_router/compiled_output.csv`
- `expected_outputs/csv_ticket_router/TCK-001.json` (and others)
- `expected_outputs/export_examples/`

## Notes

- **LLM Variations**: Outputs may vary slightly due to LLM non-determinism. Focus on structure and key content rather than exact text matches.
- **Token Counting**: Verify token counts are tracked correctly for cost estimation.
- **Progress Tracking**: Ensure progress updates in real-time for async operations.
- **Error Messages**: Verify error messages are clear and actionable.

