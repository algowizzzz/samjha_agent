# AI Bulk Doc Analysis  
## Technical Requirements Document (TRD) — Final v1

---

## 1. Product Overview

**Product Name:** AI Bulk Doc Analysis  

**Purpose:**  
AI Bulk Doc Analysis enables users to ingest large numbers of documents, configure reusable multi-step AI prompt chains, and execute those chains deterministically across all documents to produce auditable outputs.

Current supported output format: **Markdown (.md)**.

---

## 2. High-Level Architecture

The product UI is organized into **three strictly ordered panels**:

1. **Panel 1 — Documents (Ingestion & Conversion)**
2. **Panel 2 — Build Prompt Chain (Configuration)**
3. **Panel 3 — Run & Output (Execution & Download)**

---

## 3. Panel 1 — Documents (Left Panel)

### Supported Inputs
- PDF (enabled)
- Vision / OCR (coming soon)
- DOCX / TXT / MD (coming soon)

### Key Features
- Upload multiple PDFs
- Convert to Markdown
- View conversion status
- Delete errored documents to enable execution

---

## 4. Panel 2 — Build Prompt Chain (Middle Panel)

### Features
- Select saved prompt chain (name, description, step count)
- Create new chain
- Edit prompts per step
- Enforced input dependencies (R1…R(n−1))

---

## 5. Panel 3 — Run & Output (Right Panel)

### Features
- Run execution when prerequisites met
- Per-document progress
- Input/output token visibility
- Download final Markdown output

---

## 6. Future Features
- OCR / Vision ingestion
- Retry failed documents
- Run from step

---

## 7. APIs
See detailed TRD for full API definitions.

---

## 8. Final Definition

**AI Bulk Doc Analysis** is a deterministic system for bulk document ingestion, AI-driven analysis via prompt chains, and auditable output generation.
