# Backend Feature Inventory by Panel

This document translates the finalized TRD into a backend-oriented feature list, organized by UI panel. It is intended to be used by backend engineers to design APIs, data models, and processing pipelines.

---

## Panel 1 — Documents (Ingestion & Conversion)

### A) Session & Document Intake
- Create a user **session / batch** to group uploaded documents
- Support multi-file upload (PDF only for now)
- Persist document metadata:
  - `doc_id`, `session_id`, `original_filename`, `file_type`, `size_bytes`
  - lifecycle fields: `status`, `error_code`, `error_message`, timestamps

### B) PDF → Markdown Conversion
- Trigger conversion job per document or per session
- Track lifecycle: `QUEUED → PROCESSING → CONVERTED | ERROR`
- Store converted `.md` artifact
- Expose conversion status list per session

### C) Document Listing & Actions
- List all documents in a session with status and artifacts
- Allow **delete of errored documents**
- Clickable document rows → fetch document detail

### D) Future‑proofing (Coming Soon)
- Additional input types (DOCX, TXT, MD)
- OCR / Vision ingestion

---

## Panel 2 — Prompt Chain Configuration

### A) Chain Library
- List saved chains with summary metadata:
  - `chain_id`, `name`, `description`, `step_count`, `updated_at`
- Select a chain for execution

### B) Chain Management
- Create / update chain metadata
- Add, remove, reorder steps
- Persist prompt content per step
- Snapshot chain at run-time (`chain_version_id`)

### C) Dependency Model (R0…Rn)
- Enforce step input dependencies (R0 = converted doc)
- Validate chain completeness before execution

### D) Descriptions for UX
- Store overall chain description
- Optional per-step description

---

## Panel 3 — Run & Output

### A) Run Initialization
- Create a run using `session_id` + `chain_version_id`
- Validate readiness (docs converted, chain valid)

### B) Per‑Document Execution
- Sequential step execution per document
- Track per-doc, per-step status
- Persist step outputs

### C) Token Accounting
- Capture input/output tokens:
  - per step
  - per document
  - aggregated per run

### D) Output Retrieval
- Download final output per document
- Download all outputs for a run/session
- Provide summary counts (processed / failed)

### E) Error Handling
- Step-level and doc-level error reporting
- Scaffold retries and resume-from-step (future)

---

## Cross‑Panel Capabilities

### Identity & Access
- User / team scoping for sessions, chains, runs

### Unified Status Model
- Shared enums for conversion and execution states

### Auditability
- Immutable references to:
  - original files
  - converted artifacts
  - chain snapshot used
  - per-step outputs

