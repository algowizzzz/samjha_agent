# Implementation Analysis: Workflow System Enhancements

**Date:** 2025-01-XX  
**Source:** `enhancements_prod.md`  
**Current System:** 3-panel bulk doc analysis (PDF-only, chain-based)  
**Target System:** Workflow system (multi-file, domain-governed, ingestion+chain+export)

**Related Documents:**
- `IMPLEMENTATION_PLAN_FINAL.md` - 8-phase implementation plan
- `BUILD_DETAILS.md` - Detailed build instructions with code examples

---

## Executive Summary

The enhancements transform the current system from a **simple 3-panel bulk PDF processor** into a **comprehensive workflow management system** with:

1. **Workflow entity** (combines ingestion + chain + export)
2. **Admin vs Runner UI separation**
3. **Multi-file type support** (TXT, MD, PDF, DOCX, CSV)
4. **Two ingestion modes** (Programmatic vs Vision/LLM)
5. **Export configuration** (CSV, JSON, PDF, MD, DOCX)
6. **Domain-based access control**
7. **CSV special handling** (row = task)
8. **Step titles** (new UX requirement)
9. **Workflow versioning** (beyond chain versioning)

---

## 1. Database Schema Changes

### 1.1 New Tables Required

#### `workflows` (Top-Level Entity)
```sql
CREATE TABLE workflows (
    workflow_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,  -- 20-240 chars enforced
    visibility_scope VARCHAR(50) NOT NULL,  -- 'super' | 'domain'
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_workflows_created_by ON workflows(created_by);
CREATE INDEX idx_workflows_updated_at ON workflows(updated_at);
```

#### `workflow_versions` (Immutable Snapshots)
```sql
CREATE TABLE workflow_versions (
    workflow_version_id VARCHAR(255) PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    ingestion_profile_id VARCHAR(255) NOT NULL,
    chain_version_id VARCHAR(255) NOT NULL REFERENCES chain_versions(chain_version_id),
    export_profile_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workflow_id, version_number)
);

CREATE INDEX idx_workflow_versions_workflow_id ON workflow_versions(workflow_id);
CREATE INDEX idx_workflow_versions_chain_version_id ON workflow_versions(chain_version_id);
```

#### `workflow_domains` (Many-to-Many)
```sql
CREATE TABLE workflow_domains (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
    domain VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(workflow_id, domain)
);

CREATE INDEX idx_workflow_domains_workflow_id ON workflow_domains(workflow_id);
CREATE INDEX idx_workflow_domains_domain ON workflow_domains(domain);
```

#### `ingestion_profiles` (New Entity)
```sql
CREATE TABLE ingestion_profiles (
    ingestion_profile_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    accepted_input_types TEXT[] NOT NULL,  -- ['PDF', 'DOCX', 'TXT', 'MD', 'CSV']
    mode VARCHAR(50) NOT NULL,  -- 'programmatic' | 'vision'
    vision_prompt TEXT,  -- Prompt content stored in DB (JSONB or TEXT) - required if mode='vision'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_ingestion_profiles_mode ON ingestion_profiles(mode);
```

#### `export_profiles` (New Entity)
```sql
CREATE TABLE export_profiles (
    export_profile_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    format VARCHAR(50) NOT NULL,  -- 'CSV' | 'JSON' | 'MD' | 'DOCX' | 'PDF'
    config JSONB DEFAULT '{}'::jsonb,  -- Format-specific config
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_export_profiles_format ON export_profiles(format);
```

### 1.2 Schema Modifications

#### `chain_steps` - Add `title` field
```sql
ALTER TABLE chain_steps ADD COLUMN title VARCHAR(500) NOT NULL DEFAULT 'Untitled Step';
-- Update existing rows to have meaningful titles
-- Make NOT NULL after migration
```

#### `runs` - Add `workflow_version_id`
```sql
ALTER TABLE runs ADD COLUMN workflow_version_id VARCHAR(255) REFERENCES workflow_versions(workflow_version_id);
-- Make nullable initially, then make required after migration
```

#### `documents` - Support CSV container concept
```sql
-- No schema change needed, but logic change:
-- For CSV: doc_id = file, but execution creates "tasks" (rows)
-- Need new table: execution_tasks
```

#### `execution_tasks` (New - for CSV row-based execution)
```sql
CREATE TABLE execution_tasks (
    task_id VARCHAR(255) PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL,  -- CSV row number (0-indexed)
    row_data JSONB NOT NULL,  -- Serialized row data
    status VARCHAR(50) NOT NULL,  -- 'QUEUED' | 'RUNNING' | 'SUCCESS' | 'ERROR'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, doc_id, row_index)
);

CREATE INDEX idx_execution_tasks_run_id ON execution_tasks(run_id);
CREATE INDEX idx_execution_tasks_doc_id ON execution_tasks(doc_id);
CREATE INDEX idx_execution_tasks_status ON execution_tasks(status);
```

#### `step_results` - Support task_id (for CSV)
```sql
ALTER TABLE step_results ADD COLUMN task_id VARCHAR(255) REFERENCES execution_tasks(task_id);
-- For non-CSV: task_id = NULL (doc_id-based execution)
-- For CSV: task_id = row task, doc_id = CSV file container
```

---

## 2. Model Changes (SQLAlchemy)

### 2.1 New Models

```python
# external/ai_bulk_doc_analysis/models.py

class Workflow(Base):
    __tablename__ = "workflows"
    workflow_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    visibility_scope = Column(String(50), nullable=False)
    created_by = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_json = Column(JSONB().with_variant(JSON(), 'sqlite'), default={})
    
    # Relationships
    domains = relationship("WorkflowDomain", back_populates="workflow", cascade="all, delete-orphan")
    versions = relationship("WorkflowVersion", back_populates="workflow", cascade="all, delete-orphan")

class WorkflowVersion(Base):
    __tablename__ = "workflow_versions"
    workflow_version_id = Column(String(255), primary_key=True)
    workflow_id = Column(String(255), ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False)
    version_number = Column(Integer, nullable=False)
    ingestion_profile_id = Column(String(255), nullable=False)
    chain_version_id = Column(String(255), ForeignKey("chain_versions.chain_version_id"), nullable=False)
    export_profile_id = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    workflow = relationship("Workflow", back_populates="versions")
    chain_version = relationship("ChainVersion", back_populates="workflow_versions")
    runs = relationship("Run", back_populates="workflow_version")
    
    __table_args__ = (UniqueConstraint("workflow_id", "version_number", name="uq_workflow_version"),)

class WorkflowDomain(Base):
    __tablename__ = "workflow_domains"
    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(255), ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False)
    domain = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    workflow = relationship("Workflow", back_populates="domains")
    
    __table_args__ = (UniqueConstraint("workflow_id", "domain", name="uq_workflow_domain"),)

class IngestionProfile(Base):
    __tablename__ = "ingestion_profiles"
    ingestion_profile_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    accepted_input_types = Column(ARRAY(String).with_variant(Text(), 'sqlite'), nullable=False)
    mode = Column(String(50), nullable=False)  # 'programmatic' | 'vision'
    vision_prompt = Column(Text, nullable=True)  # Prompt content stored in DB (not file path)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_json = Column(JSONB().with_variant(JSON(), 'sqlite'), default={})

class ExportProfile(Base):
    __tablename__ = "export_profiles"
    export_profile_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    format = Column(String(50), nullable=False)  # 'CSV' | 'JSON' | 'MD' | 'DOCX' | 'PDF'
    config_json = Column(JSONB().with_variant(JSON(), 'sqlite'), default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

class ExecutionTask(Base):
    __tablename__ = "execution_tasks"
    task_id = Column(String(255), primary_key=True)
    run_id = Column(String(255), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False)
    doc_id = Column(String(255), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    row_index = Column(Integer, nullable=False)
    row_data = Column(JSONB().with_variant(JSON(), 'sqlite'), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    run = relationship("Run", back_populates="tasks")
    document = relationship("Document", back_populates="tasks")
    step_results = relationship("StepResult", back_populates="task")
    
    __table_args__ = (UniqueConstraint("run_id", "doc_id", "row_index", name="uq_execution_task"),)
```

### 2.2 Modified Models

```python
# chain_steps - Add title
class ChainStep(Base):
    # ... existing fields ...
    title = Column(String(500), nullable=False)  # NEW - required field

# runs - Add workflow_version_id
class Run(Base):
    # ... existing fields ...
    workflow_version_id = Column(String(255), ForeignKey("workflow_versions.workflow_version_id"), nullable=True)  # NEW

# step_results - Add task_id
class StepResult(Base):
    # ... existing fields ...
    task_id = Column(String(255), ForeignKey("execution_tasks(task_id)"), nullable=True)  # NEW
```

---

## 3. Service Layer Changes

### 3.1 New Services Required

#### `WorkflowService`
- `create_workflow(user_id, name, description, domains, ingestion_profile_id, chain_version_id, export_profile_id)`
- `update_workflow(workflow_id, ...)` → creates new version
- `list_workflows(user_id, domain_filter=None)` → domain-scoped visibility
- `get_workflow(workflow_id)`
- `get_workflow_version(workflow_version_id)`
- `delete_workflow(workflow_id)`

#### `IngestionService`
- `create_ingestion_profile(name, accepted_input_types, mode, vision_prompt_path=None)`
- `get_ingestion_profile(ingestion_profile_id)`
- `list_ingestion_profiles()`
- **NEW:** `ingest_file(file_path, ingestion_profile, doc_id)` → returns R0 content
  - Programmatic: use libraries (pdfplumber, python-docx, etc.)
  - Vision: convert to images, call Claude with vision prompt

#### `ExportService`
- `create_export_profile(name, format, config)`
- `get_export_profile(export_profile_id)`
- `list_export_profiles()`
- **NEW:** `export_results(run_id, export_profile)` → generates final output files
  - CSV: compile from JSON outputs programmatically
  - JSON: serialize structured data
  - MD/DOCX/PDF: format appropriately

### 3.2 Modified Services

#### `BulkDocService` / `BulkDocDBService`
- **Modify `create_documents()`**: Accept multiple file types, route to appropriate ingestion
- **Modify `create_run()`**: Accept `workflow_version_id` instead of just `chain_version_id`
- **NEW:** `create_csv_tasks(run_id, doc_id, csv_data)` → creates execution_tasks per row
- **Modify execution flow**: Support both document-based and task-based (CSV) execution

---

## 4. Worker Changes

### 4.1 Conversion Worker (`conversion_worker.py`)

**Current:** PDF → Markdown only  
**New:** Multi-format ingestion

```python
def convert_doc_job(job_data):
    """
    Enhanced to support:
    - Multiple file types (TXT, MD, PDF, DOCX, CSV)
    - Programmatic vs Vision mode
    - CSV special handling (creates tasks, not R0)
    """
    ingestion_profile = get_ingestion_profile(job_data["ingestion_profile_id"])
    file_type = detect_file_type(job_data["file_path"])
    
    if ingestion_profile.mode == "programmatic":
        if file_type == "PDF":
            r0_content = convert_pdf_programmatic(file_path)
        elif file_type == "DOCX":
            r0_content = convert_docx_programmatic(file_path)
        elif file_type in ["TXT", "MD"]:
            r0_content = read_text_file(file_path)
        elif file_type == "CSV":
            # Special: create tasks, not R0
            create_csv_tasks(doc_id, file_path)
            return {"status": "CSV_PROCESSED", "task_count": N}
    elif ingestion_profile.mode == "vision":
        r0_content = convert_with_vision(file_path, ingestion_profile.vision_prompt_path)
    
    # Save R0.md
    save_r0(doc_id, r0_content)
```

### 4.2 Execution Worker (`execution_worker.py`)

**Current:** Document-based execution  
**New:** Support task-based (CSV) execution

```python
def execute_step_job(job_data):
    """
    Enhanced to support:
    - task_id (for CSV row execution)
    - If task_id present: load R0 from task.row_data, not doc
    - Export step (final step): generate export format
    """
    if job_data.get("task_id"):
        # CSV row execution
        task = get_execution_task(job_data["task_id"])
        r0_content = task.row_data  # Serialized row
    else:
        # Document-based execution
        r0_content = load_r0_from_doc(job_data["doc_id"])
    
    # ... existing execution logic ...
    
    # Check if this is export step
    if is_export_step(step_index, chain):
        output = generate_export(run_id, doc_id, export_profile)
    else:
        output = call_claude(...)
```

### 4.3 New Worker: Export Worker (`export_worker.py`)

```python
def export_results_job(job_data):
    """
    Generates final export files:
    - CSV: compile from JSON outputs
    - JSON: serialize structured data
    - MD/DOCX/PDF: format appropriately
    """
    export_profile = get_export_profile(job_data["export_profile_id"])
    run = get_run(job_data["run_id"])
    
    if export_profile.format == "CSV":
        # Compile from execution_tasks outputs
        csv_data = compile_csv_from_tasks(run_id)
        save_csv(run_id, csv_data)
    elif export_profile.format == "JSON":
        json_data = compile_json_from_results(run_id)
        save_json(run_id, json_data)
    # ... other formats
```

---

## 5. API Changes

### 5.1 New Endpoints

#### Workflow Management (Admin)
- `GET /api/bulk-doc-analysis/workflows` - List workflows (domain-filtered)
- `POST /api/bulk-doc-analysis/workflows` - Create workflow
- `GET /api/bulk-doc-analysis/workflows/<id>` - Get workflow
- `PUT /api/bulk-doc-analysis/workflows/<id>` - Update workflow (creates version)
- `DELETE /api/bulk-doc-analysis/workflows/<id>` - Delete workflow
- `GET /api/bulk-doc-analysis/workflows/<id>/versions` - List versions

#### Ingestion Profiles (Admin)
- `GET /api/bulk-doc-analysis/ingestion-profiles` - List profiles
- `POST /api/bulk-doc-analysis/ingestion-profiles` - Create profile
- `GET /api/bulk-doc-analysis/ingestion-profiles/<id>` - Get profile
- `PUT /api/bulk-doc-analysis/ingestion-profiles/<id>` - Update profile
- `DELETE /api/bulk-doc-analysis/ingestion-profiles/<id>` - Delete profile

#### Export Profiles (Admin)
- `GET /api/bulk-doc-analysis/export-profiles` - List profiles
- `POST /api/bulk-doc-analysis/export-profiles` - Create profile
- `GET /api/bulk-doc-analysis/export-profiles/<id>` - Get profile
- `PUT /api/bulk-doc-analysis/export-profiles/<id>` - Update profile
- `DELETE /api/bulk-doc-analysis/export-profiles/<id>` - Delete profile

#### Runner UI (Non-Admin)
- `GET /api/bulk-doc-analysis/workflows/available` - List workflows user can run (metadata only, no prompts)
- `POST /api/bulk-doc-analysis/runs` - Create run (now accepts `workflow_version_id`)

### 5.2 Modified Endpoints

#### `POST /api/bulk-doc-analysis/documents/upload`
- **Current:** PDF only
- **New:** Accept multiple file types (validate against workflow's accepted_input_types)
- **New:** Accept `workflow_version_id` to validate file types

#### `POST /api/bulk-doc-analysis/runs`
- **Current:** Accepts `chain_version_id`
- **New:** Accepts `workflow_version_id` (which includes chain_version_id)
- **New:** For CSV workflows, creates execution_tasks instead of direct execution

#### `GET /api/bulk-doc-analysis/runs/<id>/progress`
- **New:** Support CSV workflows (show progress per row/task, not just per doc)

#### `GET /api/bulk-doc-analysis/runs/<id>/download-all`
- **New:** Generate export format based on workflow's export_profile

### 5.3 Chain Endpoints - Add Step Title

#### `POST /api/bulk-doc-analysis/chains`
- **New:** Each step must include `title` field (required)

#### `PUT /api/bulk-doc-analysis/chains/<id>`
- **New:** Each step must include `title` field (required)

---

## 6. UI Changes

### 6.1 New Admin Panel: "Workflow Builder"

**Location:** New route `/bulk-doc-analysis/admin/workflows` (admin-only)

**Features:**
1. **Workflow Metadata Form**
   - Name (3-80 chars)
   - Description (20-240 chars, required)
   - Domain(s) (multi-select, at least 1)

2. **Step 1: Ingestion & Export Setup**
   - Upload types selector (TXT, MD, PDF, DOCX, CSV)
   - Ingestion mode selector (Programmatic / Vision)
   - If Vision: Upload vision prompt (.md file)
   - Export type selector (CSV, JSON, MD, DOCX, PDF)

3. **Step 2: Prompt Chain**
   - Existing chain editor, but:
     - Each step requires **title** field
     - Step titles shown in Runner UI

4. **Workflow Management**
   - List workflows
   - Edit (creates new version)
   - Delete
   - View versions

### 6.2 Modified Runner UI: "Run Workflows"

**Location:** `/bulk-doc-analysis` (existing route, but redesigned)

**Changes:**
1. **Workflow Selection** (replaces chain selection)
   - Cards showing:
     - Workflow name
     - Domain(s)
     - Description (2-line clamp)
     - Step titles (first 3, then "+N more")
     - Input types supported
     - Ingestion mode badge
     - Export type
     - Step count
     - Last updated
   - Search by name/description
   - Filter by domain
   - Filter by input type

2. **Upload Section**
   - File picker filtered by workflow's accepted_input_types
   - Show ingestion mode info
   - For CSV: show row count preview

3. **Execution Progress**
   - For non-CSV: per-document progress (existing)
   - For CSV: per-row/task progress (new)

4. **Download**
   - Format matches workflow's export_profile
   - CSV workflows: download compiled CSV

### 6.3 Hidden in Runner UI

**Do NOT show:**
- Prompt contents
- Vision prompts
- R-selection logic
- Token budgets
- Library details
- Edit/delete controls

---

## 7. Domain & Access Control

### 7.1 User Model Changes

**Current:** `users.json` has `roles: ["admin"]`  
**New:** Need domain assignment

```json
{
  "user_id": "user1",
  "roles": ["domain_admin"],
  "domains": ["Risk", "Compliance"],  // NEW
  "tools": ["*"]
}
```

**Roles:**
- `super_admin`: See all workflows
- `domain_admin`: See workflows for assigned domains, can create workflows
- `domain_user`: See workflows for assigned domains, run-only

### 7.2 Access Control Logic

```python
def can_view_workflow(user_session, workflow):
    if is_super_admin(user_session):
        return True
    user_domains = user_session.get("domains", [])
    workflow_domains = get_workflow_domains(workflow.workflow_id)
    return bool(set(user_domains) & set(workflow_domains))

def can_edit_workflow(user_session, workflow):
    if is_super_admin(user_session):
        return True
    if "domain_admin" in user_session.get("roles", []):
        user_domains = user_session.get("domains", [])
        workflow_domains = get_workflow_domains(workflow.workflow_id)
        return bool(set(user_domains) & set(workflow_domains))
    return False
```

---

## 8. CSV Special Handling

### 8.1 Concept

- **CSV file = container** (doc_id points to file)
- **Each row = independent task** (task_id per row)
- **R0 for task = serialized row** (JSON or MD format)
- **Execution = per-row** (not per-file)
- **Export = compiled CSV** (programmatic, not LLM-generated)

### 8.2 Implementation

```python
# On CSV upload
def process_csv_upload(doc_id, csv_file_path):
    import pandas as pd
    df = pd.read_csv(csv_file_path)
    
    # Create execution_tasks for each row
    for idx, row in df.iterrows():
        task_id = str(uuid.uuid4())
        task = ExecutionTask(
            task_id=task_id,
            run_id=run_id,
            doc_id=doc_id,
            row_index=idx,
            row_data=row.to_dict()  # Serialized row
        )
        db.add(task)
    
    # Document status = "CONVERTED" (no R0.md created)
    # Tasks are QUEUED for execution

# During execution
def execute_csv_task(task_id, step_index, workflow_config):
    task = get_execution_task(task_id)
    
    # R0 can be MD or JSON based on workflow config
    # JSON structure is defined by the prompt (LLM outputs structured JSON)
    if workflow_config.csv_r0_format == "json":
        r0_content = json.dumps(task.row_data, indent=2)
    else:  # markdown
        r0_content = format_row_as_markdown(task.row_data)
    
    # ... execute step with r0_content as R0 ...
    # If prompt asks for JSON output, LLM will return structured JSON
    
# On export
def export_csv_results(run_id):
    tasks = get_all_tasks_for_run(run_id)
    results = []
    for task in tasks:
        final_output = get_final_step_output(task_id)
        results.append({
            **task.row_data,  # Original row
            **parse_output_to_dict(final_output)  # LLM outputs
        })
    df = pd.DataFrame(results)
    return df.to_csv()
```

---

## 9. Vision Ingestion

### 9.1 Process

1. **Upload vision prompt** (text/markdown) during workflow creation
2. **Store prompt** in `ingestion_profiles.vision_prompt` (database, not file)
3. **During ingestion (PDF only):**
   - Convert PDF pages to images (PNG) - including images/visuals in PDF
   - Base64 encode images
   - Call Claude Vision API with prompt + images
   - Extract text/markdown from response
   - Merge into R0.md

### 9.2 Implementation

```python
def convert_with_vision(file_path, vision_prompt):
    """
    Convert PDF using Claude Vision API.
    PDF only - includes images and visuals in the PDF.
    """
    # Validate PDF only
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError("Vision ingestion only supports PDF files")
    
    # Convert PDF pages to images (PNG)
    images = pdf_to_images(file_path)  # Returns list of PIL Images or bytes
    
    # Call Claude Vision API
    messages = []
    for img in images:
        img_base64 = base64_encode(img)
        messages.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_base64}
        })
    
    messages.append({
        "type": "text",
        "text": vision_prompt  # From database, not file
    })
    
    response = claude_client.messages.create(
        model="claude-3-opus-20240229",  # Vision-capable model
        messages=[{"role": "user", "content": messages}]
    )
    
    return response.text  # R0 content
```

---

## 10. Export Generation

### 10.1 Export Types

#### CSV Export
- Compile from `execution_tasks` outputs
- Original row data + LLM outputs
- Programmatic (pandas)

#### JSON Export
- Serialize structured data from step results
- Include metadata (run_id, doc_id, timestamps)

#### MD Export
- Current behavior (R(N).md files)

#### DOCX Export
- Use `python-docx` to format markdown content

#### PDF Export
- Use `reportlab` or convert DOCX → PDF

### 10.2 Implementation

```python
def generate_export(run_id, export_profile):
    run = get_run(run_id)
    workflow_version = get_workflow_version(run.workflow_version_id)
    
    if export_profile.format == "CSV":
        return export_csv(run_id)
    elif export_profile.format == "JSON":
        return export_json(run_id)
    elif export_profile.format == "MD":
        return export_markdown(run_id)  # Current behavior
    elif export_profile.format == "DOCX":
        return export_docx(run_id)
    elif export_profile.format == "PDF":
        return export_pdf(run_id)
```

---

## 11. Token Counting Enhancements

### 11.1 Ingestion-Time Estimation

```python
def estimate_ingestion_tokens(file_path, ingestion_profile):
    if ingestion_profile.mode == "programmatic":
        r0_content = convert_programmatic(file_path)
        return count_tokens(r0_content)
    elif ingestion_profile.mode == "vision":
        # Estimate: prompt tokens + image payload estimate
        vision_prompt = load_vision_prompt(ingestion_profile.vision_prompt_path)
        prompt_tokens = count_tokens(vision_prompt)
        # Rough estimate: ~1000 tokens per image (adjust based on size)
        image_count = get_page_count(file_path)
        estimated_tokens = prompt_tokens + (image_count * 1000)
        return estimated_tokens
```

### 11.2 Display in UI

- Show estimated tokens before upload
- Show actual tokens after ingestion
- Show token usage per step (existing)

---

## 12. Migration Strategy

### 12.1 Data Migration

1. **Existing chains → workflows**
   - Create default workflow for each chain
   - Create default ingestion_profile (PDF, programmatic)
   - Create default export_profile (MD)
   - Link via workflow_version

2. **Existing runs**
   - Add `workflow_version_id` (nullable initially)
   - Backfill from `chain_version_id`

3. **Chain steps**
   - Add `title` column (default: "Step {index}")
   - Allow admins to update titles

### 12.2 Backward Compatibility

- Keep existing chain endpoints working
- Auto-create workflows for existing chains (lazy migration)
- Support both `chain_version_id` and `workflow_version_id` in runs (transition period)

---

## 13. Dependencies & Libraries

### 13.1 New Python Packages Required

```txt
# Ingestion
pdfplumber>=0.10.0  # Already have
python-docx>=1.1.0  # NEW
PyMuPDF>=1.23.0  # Alternative PDF library
pandas>=2.0.0  # For CSV handling

# Vision
Pillow>=10.0.0  # Image processing
pdf2image>=1.16.0  # PDF to images

# Export
python-docx>=1.1.0  # DOCX generation
reportlab>=4.0.0  # PDF generation
pandas>=2.0.0  # CSV compilation

# Existing
rq>=1.15.0  # Already have
redis>=5.0.0  # Already have
sqlalchemy>=2.0.0  # Already have
```

---

## 14. Testing Requirements

### 14.1 Unit Tests

- Workflow CRUD operations
- Ingestion profiles (programmatic vs vision)
- Export generation (all formats)
- CSV task creation and execution
- Domain access control logic

### 14.2 Integration Tests

- End-to-end workflow: create → run → export
- CSV workflow: upload → task creation → execution → CSV export
- Vision ingestion: PDF → images → Claude → R0
- Domain filtering: user sees only allowed workflows

### 14.3 Performance Tests

- CSV workflows with 1000+ rows
- Vision ingestion with large PDFs
- Bulk export generation

---

## 15. Documentation Updates

### 15.1 User Documentation

- Admin guide: Creating workflows
- Runner guide: Executing workflows
- CSV workflow guide
- Vision ingestion guide

### 15.2 API Documentation

- New endpoints
- Workflow data models
- Domain access control rules

---

## 16. Clarification Questions - ANSWERED

### 16.1 Domain System ✅

**Q1:** Does a domain system already exist in the codebase, or do we need to build it from scratch?
**A:** **Build from scratch** - No existing domain system

**Q2:** Should domains be:
**A:** **To be determined** - Will implement as free-form strings initially (user types "Risk", "Compliance", etc.) for simplicity. Can enhance later.

**Q3:** How should domain assignment work?
**A:** **Admin assigns domains to users** - Extend `users.json` with `domains[]` field. Can add domain management UI later if needed.

### 16.2 Vision Ingestion ✅

**Q4:** For vision ingestion, should we:
**A:** **PDF only** - Including images and visuals in PDF. Convert PDF pages to images.

**Q5:** Vision prompt storage:
**A:** **Store content in database (JSONB)** - Store prompt text directly in `ingestion_profiles.vision_prompt` (JSONB column), not file path.

### 16.3 CSV Handling ✅

**Q6:** For CSV workflows, should R0 be:
**A:** **MD or JSON** - Configurable. JSON structure will be defined by the prompt (LLM outputs structured JSON based on prompt instructions).

**Q7:** CSV export: Should we:
**A:** **Yes** - Include original row data + LLM outputs (confirmed)

### 16.4 Export Formats ✅

**Q8:** For DOCX/PDF export, should we:
**A:** **Yes** - Export each document's final output as separate file (confirmed)

**Q9:** JSON export structure:
**A:** **Yes** - Array of objects (one per doc/task) (confirmed)

### 16.5 Workflow Versioning

**Q10:** When editing a workflow:
**A:** **To be determined** - Will implement always create new version (immutable) for consistency.

**Q11:** Version numbering:
**A:** **Auto-increment (v1, v2, v3...)** - Simple and clear.

### 16.6 UI/UX ✅

**Q12:** Admin workflow builder:
**A:** **To be determined** - Will design based on complexity. Likely multi-step wizard for better UX.

**Q13:** Runner UI workflow cards:
**A:** **Card layout (grid)** - Confirmed preference.

### 16.7 Backward Compatibility ✅

**Q14:** Should we:
**A:** **Remove old endpoints immediately** - No legacy code needed. Clean break.

**Q15:** Existing runs:
**A:** **Remove old endpoints** - No migration needed since we're removing legacy code.

---

## 16.8 Scope Boundaries ✅

**CRITICAL CONSTRAINT:**
- **DO NOT modify** anything in `external/agent/` (Parquet Agent)
- **DO NOT modify** anything in `external/tools/parquet_agent/`
- **ONLY modify** `external/ai_bulk_doc_analysis/` and related routes/UI
- Keep changes isolated to bulk document analysis feature

---

## 17. Implementation Priority

### Phase 1: Core Workflow System (MVP)
1. Database schema (workflows, workflow_versions, ingestion_profiles, export_profiles)
2. Workflow CRUD APIs
3. Basic workflow service layer
4. Admin UI: Workflow builder (basic)
5. Runner UI: Workflow selection (basic)

### Phase 2: Multi-File Support
1. Ingestion service (programmatic for TXT, MD, PDF, DOCX)
2. Modified conversion worker
3. File type validation in upload
4. UI: Multi-file type upload

### Phase 3: Domain & Access Control
1. Domain system (if not exists)
2. User domain assignment
3. Workflow domain assignment
4. Access control logic
5. UI: Domain filtering

### Phase 4: CSV Support
1. Execution tasks table
2. CSV ingestion logic
3. Task-based execution
4. CSV export generation
5. UI: CSV workflow handling

### Phase 5: Vision Ingestion
1. Vision prompt storage
2. PDF/DOCX to images conversion
3. Claude Vision API integration
4. Vision ingestion worker
5. UI: Vision prompt upload

### Phase 6: Export System
1. Export profiles
2. Export generation (all formats)
3. Export worker
4. UI: Export format selection

### Phase 7: Step Titles & UX Polish
1. Step title field (database + UI)
2. Runner UI: Step titles display
3. Workflow card polish
4. Search & filtering

### Phase 8: Token Counting & Cost Estimation
1. Ingestion-time token estimation
2. Cost preview in UI
3. Enhanced token tracking

---

## 18. Estimated Effort

| Phase | Components | Estimated Time |
|-------|-----------|----------------|
| Phase 1 | Core workflow system | 2-3 weeks |
| Phase 2 | Multi-file support | 1-2 weeks |
| Phase 3 | Domain & access control | 1-2 weeks |
| Phase 4 | CSV support | 2-3 weeks |
| Phase 5 | Vision ingestion | 2-3 weeks |
| Phase 6 | Export system | 2-3 weeks |
| Phase 7 | Step titles & UX | 1 week |
| Phase 8 | Token counting | 1 week |
| **Total** | | **12-18 weeks** |

---

## 19. Risk Assessment

### High Risk
- **Vision ingestion cost**: Large PDFs → many images → high token costs
- **CSV scalability**: 1000+ row CSVs may strain system
- **Domain system**: If building from scratch, complexity increases

### Medium Risk
- **Export format quality**: DOCX/PDF generation may need tuning
- **Backward compatibility**: Migration of existing data/runs
- **UI complexity**: Admin vs Runner separation needs clear UX

### Low Risk
- **Multi-file ingestion**: Well-established libraries
- **Workflow versioning**: Similar to existing chain versioning
- **Token counting**: Straightforward implementation

---

## 20. Next Steps

1. ✅ **Review this document** with stakeholders - DONE
2. ✅ **Answer clarification questions** (Section 16) - DONE
3. ✅ **Prioritize phases** based on business needs - DONE (see IMPLEMENTATION_PLAN_FINAL.md)
4. ✅ **Create detailed technical specs** for Phase 1 - DONE (see BUILD_DETAILS.md)
5. **Set up development environment** (new dependencies)
6. **Begin Phase 1 implementation**

---

## 21. Current Implementation Review

### 21.1 Chain Editor in Admin Panel

**Location:** `web/templates/admin.html` (lines ~1100-1632)

**Current Features:**
- Modal-based editor (`chain-editor-modal`)
- Chain name (required), description (optional)
- Steps with:
  - Required inputs (checkboxes: R0, R1, R2...)
  - Model config (model, max_tokens, temperature)
  - Prompt (textarea, required)
  - Description (optional)
- Create/Edit modes
- Validation (name required, at least 1 step, each step needs prompt)

**Current API Endpoints:**
- `GET /api/bulk-doc-analysis/chains` - List chains
- `POST /api/bulk-doc-analysis/chains` - Create chain
- `PUT /api/bulk-doc-analysis/chains/<chain_id>` - Update chain

**Current Data Structure:**
```javascript
chainEditorState = {
    mode: 'create' | 'edit',
    chainId: string | null,
    name: string,
    description: string,
    steps: [{
        index: number,
        required_inputs: string[],  // ['R0', 'R1', ...]
        prompt: string,
        description: string,
        model_config: {
            model: string,
            max_tokens: number,
            temperature: number
        }
    }]
}
```

**What Needs to Change:**
1. Add `title` field to each step (required)
2. Replace chain editor with workflow editor (3-step wizard)
3. Add ingestion & export configuration
4. Add domain selection
5. Remove old chain endpoints

**See BUILD_DETAILS.md Section 4.1 for detailed UI implementation.**

---

**End of Analysis**

