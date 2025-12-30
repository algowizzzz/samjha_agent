Yes — it makes sense, and it’s a clean evolution of the current 3-panel “deterministic bulk pipeline” into **(1) Admin workflow builder** + **(2) Runner UI** while keeping the same execution semantics (R0…Rn, per-step audit trail, pinned chain versions, token tracking).   

Below is how I’d structure it so FE/BE stays simple and you don’t break the existing mental model. 

---

## 1) New UI Information Architecture

### A) Admin: “Workflow Builder” (new area)

A **workflow = ingestion pipeline + prompt chain + export step**.

**Step 1 — Ingestion & Export Setup**

* **Upload types (input):** txt, md, pdf, docx, csv
* **Ingestion mode (per type):**

  * **Programmatic extract (no images)** for pdf/docx (tables included)
  * **Vision (LLM)** for pdf/docx/images *requires prompt upload (.md)*
  * **CSV ingestion:** each row = one task (special pipeline)
* **Export type:** csv, json, pdf, md, docx (final step is fixed and programmatic)

**Step 2 — Prompt Chain**

* Name
* Domain (multi-domain)
* Steps (same as before: R0→R1…Rn; each step has prompt .md and input selection rules)

**CRUD**

* Save / Edit / Delete workflows
* Versioning: editing creates a new immutable version (so old runs remain reproducible)  

### B) Non-Admin: “Run Workflows”

* If **super admin**: see all workflows
* If **domain admin/user**: see workflows allowed for their domain(s)
* Actions:

  * Choose workflow
  * Upload docs
  * Run
  * Download results (per doc + bulk) 

This preserves the “deterministic feel” and avoids turning the product into a chat UI. 

---

## 2) Key Backend Model Changes

### Workflow entity (new “top-level” object)

A `workflow` references:

* `ingestion_profile_id` (input type + mode + optional vision prompt)
* `chain_version_id` (prompt chain snapshot)
* `export_profile_id` (export type + config)
* `domains[]`
* `visibility_scope` (super/global vs domain-scoped)

This keeps your existing **chain versioning** intact and just composes it into a bigger reusable unit. 

### CSV: “row = task” execution shape

Treat CSV as:

* `doc_id` = the CSV file (container)
* `task_id` = each row
* `R0` for a task = serialized row payload (json or md)
* Execution produces `Rn` per row
* **CSV export** = compiled programmatically from row outputs (not “LLM generates CSV”)

This fits your “one step = one Claude call” rule while scaling cleanly.  

---

## 3) Ingestion & Export Libraries (practical defaults)

### Ingestion (recommended)

* **PDF programmatic (no images):** `pdfplumber` or `PyMuPDF` for text + tables; optionally `camelot/tabula` for tables when needed.
* **DOCX:** `python-docx`
* **TXT/MD:** direct read (MD preserved as-is; treat as already-normalized R0.md)
* **CSV:** `pandas` / `csv` module

### Vision ingestion (LLM)

* Convert pdf → images per page (png), base64 encode, call Claude with the uploaded vision prompt.
* Store:

  * per-page extracted text/markdown
  * merged R0.md (or R0.json) as the normalized document for downstream steps

### Export (fixed final step)

* **MD/TXT:** direct
* **JSON:** structured serialize
* **CSV:** compile from stored JSON outputs deterministically
* **DOCX:** `python-docx`
* **PDF:** `reportlab` (or docx→pdf in a controlled service if you prefer)

This aligns with your existing worker boundary (API enqueues, workers do conversion + Claude calls + artifact building). 

---

## 4) Token Counting (where it belongs)

Add token accounting at two places:

1. **Upload time:** estimate tokens for normalized R0 (helps cost preview)
2. **Run time:** actual in/out tokens per step per doc/task (already in your execution contract)  

For programmatic ingestion, token estimate = token count of produced R0.
For vision ingestion, estimate can be:

* prompt tokens + page payload size approximation (and then replace with actual after call).

---

## 5) UX specifics to keep it simple

* In **Runner UI**, hide complexity:

  * Workflow card shows: input types supported, ingestion mode (programmatic/vision), export type, step count, last updated, domains.
* Disable upload types that workflow doesn’t accept (no “try and fail”).
* For CSV workflows, show: “rows detected = N tasks” and progress per row batch, not per file. 

---

## 6) A few “gotchas” to lock down now

* **“Ignore images” promise:** enforce it by default in programmatic mode; do not accidentally pass embedded images to LLM.
* **Tables in PDFs:** decide “best-effort extraction” vs “strict table fidelity,” and surface a warning badge if table extraction confidence is low.
* **Vision prompt required:** validate at workflow save-time (not at runtime).
* **Version pinning:** runs must pin `workflow_version_id` (which pins ingestion+chain+export) just like chain pinning today. 

---
Good question — this is purely a **UX abstraction**, not a backend change.

What I meant is: **the Runner UI should not expose the internal pipeline configuration**, only a **concise, decision-oriented summary** so a user can confidently pick the right workflow without opening or understanding the admin setup.

Below is the precise meaning, field by field.

---

## “Workflow card” — what it is

In the **non-admin panel (Run Workflows)**, each saved prompt chain (workflow) is rendered as a **card or row** — similar to a “job template”.

The card is **read-only metadata**, not editable here.

---

## What each item means (concretely)

### 1) **Input types supported**

**Purpose:** Prevents user error at upload time.

**Displayed as:**

```
Inputs: PDF · DOCX · MD · CSV
```

**Backend source:**

* From `ingestion_profile.accepted_input_types[]`

**UI behavior:**

* File picker only allows these extensions
* Unsupported files are blocked before upload

---

### 2) **Ingestion mode**

**Purpose:** Sets user expectation about cost, speed, and accuracy.

**Displayed as (badge):**

```
Ingestion: Programmatic
```

or

```
Ingestion: Vision (LLM)
```

**Backend source:**

* `ingestion_profile.mode = programmatic | vision`

**Why this matters:**

* Programmatic → faster, cheaper, ignores images
* Vision → slower, costlier, uses uploaded vision prompt

No prompt editing shown here — just the fact that vision is involved.

---

### 3) **Export type**

**Purpose:** Tells the user what they’ll get at the end.

**Displayed as:**

```
Output: Markdown
```

or

```
Output: CSV (compiled)
```

**Backend source:**

* `export_profile.format`

**UI behavior:**

* Download button label matches this
* No format choice here — fixed by workflow

---

### 4) **Step count**

**Purpose:** Communicates pipeline depth and runtime intuition.

**Displayed as:**

```
Steps: 4
```

**Backend source:**

* `prompt_chain.steps.length`

**Why this matters:**

* 1–2 steps → fast transform
* 5–8 steps → heavy reasoning / review workflow

No R0/R1/R2 mechanics shown — just the count.

---

### 5) **Last updated**

**Purpose:** Signals freshness and version trust.

**Displayed as:**

```
Updated: 3 days ago
```

**Backend source:**

* `workflow_version.created_at`

Important for regulated / enterprise usage where stale workflows matter.

---

### 6) **Domains**

**Purpose:** Explains *why* the user can see this workflow.

**Displayed as:**

```
Domains: Risk · Legal · Compliance
```

**Backend source:**

* `workflow.domains[]`

**UI behavior:**

* If user is domain-scoped, they only see workflows overlapping their domains
* Super admin sees all

---

## What is intentionally **hidden** in Runner UI

The Runner **does NOT show**:

* Prompt text
* Vision prompt
* R-selection logic
* Token budgets
* Ingestion libraries
* Export library details
* Chain edit controls

Those belong **only in Admin Workflow Builder**.

---

## Why this matters (design principle)

This keeps the Runner UI:

* **Safe** (no misconfiguration)
* **Fast** (choose → upload → run)
* **Enterprise-friendly** (template-driven, auditable)
* **Non-intimidating** for non-technical users

Think of it like:

> *“Choosing a saved report”*
> not
> *“Building a data pipeline.”*

---

## One-line summary

> A **workflow card** is a read-only, human-friendly summary that tells a user **what files they can upload, how processing happens, and what they’ll get back** — without exposing any internal pipeline mechanics.

Perfect — that’s a strong UX improvement. Here’s the clean way to implement it (Admin + Runner), with minimal backend/FE complexity.

---

## 1) Admin: New workflow fields

### Add these fields when creating/editing a workflow

* **Workflow Name** *(required, unique within a domain or within tenant — your choice)*
* **Domain(s)** *(required; multi-select)*
* **Workflow Description** *(required or strongly recommended; I’d make it required with a short limit)*

**Validation defaults**

* Name: 3–80 chars
* Description: 20–240 chars (forces “why this exists”)
* Domain: at least 1

**Storage**

* Put `description` at the workflow (or workflow_version) level:

  * If you expect description to evolve with versions → store on `workflow_version`
  * If description is stable marketing-style copy → store on `workflow`

---

## 2) Prompt chain: Step titles (explicit)

Each step already exists; add/ensure:

* **Step Title** *(required)*
  Example: “Extract policy sections”, “Check MUST/SHOULD”, “Generate redlines”, etc.

This becomes the user-friendly label for cards and progress UI.

---

## 3) Runner UI: Card content (what to show)

### Card must display

1. **Workflow Name** (primary)
2. **Domain(s)** (chips/badges)
3. **Description** (2-line clamp)
4. **Prompt chain step titles** (compact)

#### Suggested step title display rules

* Show first **3 step titles** inline
* If more than 3: show “+N more”
* On hover/tap: tooltip/popover listing all step titles

Example card:

* **Policy Review — Basel CAR**
* Domains: Risk, Compliance
* *“Extracts key obligations, checks MUST vs SHOULD, then produces a markdown report with gaps + recommendations.”*
* Steps: Extract obligations → Classify requirements → Generate gaps **(+2 more)**

---

## 4) Filtering and search (small addition, big UX win)

In Runner:

* Search box searches **name + description**
* Filters by **domain**
* Optional: “input type supported” filter (PDF/CSV/etc.)

---

## 5) Backend contract (minimal shape)

Add to the workflow list endpoint payload:

* `workflow_id`
* `workflow_version_id`
* `name`
* `domains[]`
* `description`
* `steps[]: [{ step_id, title, order }]`
* (already useful metadata): `accepted_input_types[]`, `ingestion_mode`, `export_type`, `updated_at`

No need to send full prompts to Runner.

---

If you want, I can write the exact UI copy + field help-text (so admins write consistent descriptions), and the JSON schema for both:

* **Admin create/edit workflow**
* **Runner list workflows**


Below is the **final consolidated specification document**, incorporating **all enhancements discussed**.
This is written to be directly usable for **engineering handoff + UX implementation**.

---

# Prompt Chain Workflow System

**Final Product & Technical Specification**

---

## 1. Purpose & Vision

The system enables **reusable, deterministic, AI-powered workflows** for bulk document processing.
Each workflow encapsulates:

1. **Ingestion pipeline** (how files are read & normalized)
2. **Prompt chain** (LLM reasoning steps)
3. **Export pipeline** (final output format)

There are **two distinct experiences**:

* **Admin Panel** → Define, version, and govern workflows
* **Runner UI** → Execute approved workflows safely and consistently

The design prioritizes:

* Determinism over chat
* Auditability and reproducibility
* Clear separation of concerns
* Enterprise-safe UX (no accidental misconfiguration)

---

## 2. Core Concepts

### 2.1 Workflow (Top-Level Entity)

A **workflow** represents a complete, reusable AI pipeline.

A workflow consists of:

* Metadata (name, domain, description)
* Ingestion configuration
* Prompt chain (versioned)
* Export configuration

A workflow is **immutable once versioned**.
Editing creates a **new workflow version**.

---

## 3. Admin Panel — Workflow Builder

### 3.1 Workflow Metadata (Required)

When creating a new workflow, the admin must provide:

| Field                    | Description                                  |
| ------------------------ | -------------------------------------------- |
| **Workflow Name**        | Human-readable, unique within domain/tenant  |
| **Domain(s)**            | One or more domains (multi-domain supported) |
| **Workflow Description** | Short explanation of purpose and output      |

**Guidelines**

* Name: 3–80 characters
* Description: 20–240 characters (forces clarity)
* Domain: at least one required

> The description is **mandatory** and used verbatim in the Runner UI.

---

### 3.2 Step 1 — Ingestion & Export Configuration

#### 3.2.1 Supported Upload Types

* TXT
* Markdown (MD)
* PDF
* Word (DOCX)
* CSV

---

#### 3.2.2 Ingestion Mode

Each workflow selects **one ingestion mode**:

##### A) Programmatic Extraction (Default)

* Uses libraries to extract:

  * Text
  * Tables
* **Images ignored**
* Deterministic
* Lower cost, faster

Applies to:

* PDF
* DOCX

##### B) Vision (LLM-Based)

* Requires **Vision Prompt (.md)** upload
* Converts pages to images (PNG)
* Encodes images (Base64 or equivalent)
* Sends images + prompt to LLM
* Output normalized to MD or JSON

Used when:

* Layout understanding is required
* Images or scanned PDFs matter

**Validation**

* Vision prompt is mandatory if Vision mode is selected
* Workflow cannot be saved without it

---

#### 3.2.3 CSV Ingestion (Special Pipeline)

CSV workflows are handled differently:

* Each **row = one independent task**
* The file itself is a container, not a single R0
* R0 for each task = serialized row (JSON or MD)

Execution:

* Prompt chain runs **per row**
* Outputs stored per row

CSV Export:

* Results compiled **programmatically**
* LLM does **not** generate CSV directly

---

#### 3.2.4 Token Counting (Ingestion-Time)

At upload time:

* System estimates token count for:

  * Programmatic extraction → tokens of normalized R0
  * Vision ingestion → prompt tokens + page payload estimate

Actual token usage is recorded during execution.

---

#### 3.2.5 Export Configuration (Final Fixed Step)

Export type is **fixed per workflow**:

Supported:

* CSV
* JSON
* Markdown
* Word (DOCX)
* PDF

Export is:

* Deterministic
* Programmatic
* Uses standard libraries (no LLM reasoning)

---

## 4. Step 2 — Prompt Chain Definition

### 4.1 Prompt Chain Structure

A prompt chain consists of ordered steps:

* R0 = normalized ingestion output
* R1…Rn = LLM outputs per step

Each step requires:

* **Step Title** (mandatory)
* **Prompt file (.md)** (mandatory)
* **Input selection** (R0, R1…Rn-1; multi-select allowed)

---

### 4.2 Step Titles (Critical UX Feature)

Step titles are:

* Human-readable
* Shown in Runner UI
* Used in progress tracking

Examples:

* “Extract policy sections”
* “Identify MUST vs SHOULD”
* “Generate compliance gaps”
* “Produce executive summary”

---

### 4.3 Chain Versioning

* Editing a chain creates a **new immutable version**
* All executions pin:

  * Workflow version
  * Chain version
  * Ingestion config
  * Export config

This ensures **full reproducibility**.

---

## 5. Runner UI — AI-Assisted Workflows

The Runner UI is **execution-only**.

No configuration.
No prompt editing.
No ingestion decisions.

---

### 5.1 Workflow Visibility Rules

| User Role    | Visibility                      |
| ------------ | ------------------------------- |
| Super Admin  | All workflows                   |
| Domain Admin | Workflows for assigned domains  |
| Domain User  | Same as Domain Admin (run-only) |

If Domain Admin clicks **“Create New”**, they are redirected to Admin Panel.

---

### 5.2 Workflow Card (Primary UI Element)

Each workflow is displayed as a **card (or row)** with **read-only metadata**.

#### Displayed Fields (Mandatory)

1. **Workflow Name** (primary title)
2. **Domain(s)** (chips/badges)
3. **Workflow Description** (2-line clamp)
4. **Prompt Chain Step Titles**
5. **Input Types Supported**
6. **Ingestion Mode**
7. **Export Type**
8. **Step Count**
9. **Last Updated**

---

### 5.3 Step Title Display Rules

* Show first **3 step titles**
* If more exist → show “+N more”
* Hover / tap reveals full list

Example:

```
Steps:
Extract obligations → Classify requirements → Generate gaps (+2 more)
```

---

### 5.4 What Is Explicitly Hidden in Runner UI

The Runner UI **does NOT show**:

* Prompt contents
* Vision prompts
* R-selection logic
* Token budgets
* Libraries used
* Edit or delete controls

This prevents:

* Misconfiguration
* Prompt leakage
* Accidental workflow drift

---

## 6. Runner Execution Flow

1. User selects workflow
2. Uploads allowed file types only
3. System validates:

   * File type compatibility
   * CSV row counts (if applicable)
4. Execution starts
5. Progress tracked:

   * Per file (non-CSV)
   * Per row batch (CSV)
6. Outputs available:

   * Per file / per row
   * Bulk download

---

## 7. Backend Model Summary

### 7.1 Core Entities

* `workflow`
* `workflow_version`
* `ingestion_profile`
* `prompt_chain_version`
* `prompt_step`
* `export_profile`
* `execution_run`
* `execution_task` (CSV rows)

---

### 7.2 Workflow List API (Runner UI)

Payload includes:

* workflow_id
* workflow_version_id
* name
* description
* domains[]
* accepted_input_types[]
* ingestion_mode
* export_type
* step_titles[]
* step_count
* last_updated

**No prompts returned.**

---

## 8. Key Design Principles (Locked)

1. **Admin builds, Runner executes**
2. **Workflows are templates, not chats**
3. **CSV is task-based, not file-based**
4. **Vision is explicit and intentional**
5. **Export is deterministic**
6. **Version pinning everywhere**
7. **UX hides complexity, not capability**

---

## 9. One-Line System Definition

> This system provides versioned, domain-governed, AI workflows that transform documents through deterministic ingestion, structured prompt chains, and fixed exports — safely, repeatably, and at scale.

---




