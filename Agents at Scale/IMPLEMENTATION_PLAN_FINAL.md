# Final Implementation Plan: Workflow System

**Status:** Ready to Start  
**Date:** 2025-01-XX  
**Scope:** Bulk Document Analysis Only (DO NOT touch Parquet Agent)

---

## ✅ Decisions Confirmed

1. **Domain System:** Build from scratch (free-form strings initially)
2. **Vision Ingestion:** PDF only, including images/visuals, prompt stored in DB
3. **CSV R0 Format:** MD or JSON (configurable), JSON structure defined by prompt
4. **Export Formats:** Separate files per document, JSON as array of objects
5. **UI Layout:** Card layout (grid) for Runner UI
6. **Backward Compatibility:** Remove old endpoints, no legacy code
7. **Scope:** Only modify `external/ai_bulk_doc_analysis/`, do NOT touch `external/agent/`

---

## Implementation Phases

### Phase 1: Core Workflow System (Week 1-3)

**Goal:** Basic workflow CRUD + database schema

**Tasks:**
1. Create database migration for new tables:
   - `workflows`
   - `workflow_versions`
   - `workflow_domains`
   - `ingestion_profiles` (with `vision_prompt` TEXT column)
   - `export_profiles`
   - `execution_tasks`
   - Modify: `chain_steps` (add `title`), `runs` (add `workflow_version_id`), `step_results` (add `task_id`)

2. Create SQLAlchemy models in `models.py`

3. Create `WorkflowService`:
   - `create_workflow()`
   - `update_workflow()` → creates version
   - `list_workflows()` → domain-filtered
   - `get_workflow()`
   - `delete_workflow()`

4. Create basic workflow APIs:
   - `GET /api/bulk-doc-analysis/workflows`
   - `POST /api/bulk-doc-analysis/workflows`
   - `GET /api/bulk-doc-analysis/workflows/<id>`
   - `PUT /api/bulk-doc-analysis/workflows/<id>`
   - `DELETE /api/bulk-doc-analysis/workflows/<id>`

5. Remove old chain endpoints (clean break)

**Deliverable:** Workflows can be created/edited/deleted via API

---

### Phase 2: Domain System (Week 4-5)

**Goal:** Domain assignment and access control

**Tasks:**
1. Extend `users.json` schema:
   ```json
   {
     "user_id": "user1",
     "roles": ["domain_admin"],
     "domains": ["Risk", "Compliance"],  // NEW
     ...
   }
   ```

2. Update `AuthManager`:
   - Add `get_user_domains(user_id)` method
   - Add `is_super_admin(session)` method
   - Add `is_domain_admin(session)` method

3. Create domain access control logic:
   - `can_view_workflow(user_session, workflow)`
   - `can_edit_workflow(user_session, workflow)`
   - `can_run_workflow(user_session, workflow)`

4. Update workflow APIs to enforce domain filtering

5. Create domain assignment UI (admin panel)

**Deliverable:** Users see only workflows for their domains

---

### Phase 3: Ingestion System (Week 6-8)

**Goal:** Multi-file type support + vision ingestion

**Tasks:**
1. Create `IngestionService`:
   - `create_ingestion_profile()`
   - `get_ingestion_profile()`
   - `list_ingestion_profiles()`
   - `ingest_file()` → returns R0 content

2. Implement programmatic ingestion:
   - PDF: `pdfplumber` (existing)
   - DOCX: `python-docx`
   - TXT/MD: direct read
   - CSV: special handling (creates tasks)

3. Implement vision ingestion (PDF only):
   - PDF → images conversion (`pdf2image` or `PyMuPDF`)
   - Base64 encoding
   - Claude Vision API integration
   - Store prompt in DB (not file)

4. Update conversion worker:
   - Route to programmatic vs vision based on profile
   - Handle multiple file types

5. Create ingestion profile APIs:
   - `GET /api/bulk-doc-analysis/ingestion-profiles`
   - `POST /api/bulk-doc-analysis/ingestion-profiles`
   - `PUT /api/bulk-doc-analysis/ingestion-profiles/<id>`
   - `DELETE /api/bulk-doc-analysis/ingestion-profiles/<id>`

6. Update upload endpoint:
   - Accept multiple file types
   - Validate against workflow's accepted_input_types

**Deliverable:** Can ingest TXT, MD, PDF, DOCX (programmatic) and PDF (vision)

---

### Phase 4: CSV Support (Week 9-11)

**Goal:** CSV row-based execution

**Tasks:**
1. Implement CSV task creation:
   - Parse CSV file
   - Create `execution_tasks` per row
   - Store row data as JSONB

2. Update execution worker:
   - Support `task_id` in job data
   - Load R0 from `task.row_data` (not doc)
   - Format R0 as MD or JSON based on config

3. Update run creation:
   - Detect CSV workflow
   - Create tasks instead of direct execution
   - Queue task-based jobs

4. Update progress tracking:
   - Show per-row/task progress for CSV
   - Aggregate task statuses

5. CSV export generation:
   - Compile from task outputs
   - Original row + LLM outputs

**Deliverable:** CSV workflows work end-to-end

---

### Phase 5: Export System (Week 12-14)

**Goal:** Multi-format export generation

**Tasks:**
1. Create `ExportService`:
   - `create_export_profile()`
   - `get_export_profile()`
   - `list_export_profiles()`
   - `export_results()` → generates files

2. Implement export formats:
   - CSV: compile from tasks (pandas)
   - JSON: serialize structured data (array of objects)
   - MD: current behavior (R(N).md files)
   - DOCX: format markdown (`python-docx`)
   - PDF: generate from content (`reportlab`)

3. Create export worker:
   - `export_results_job()` → generates export files

4. Create export profile APIs:
   - `GET /api/bulk-doc-analysis/export-profiles`
   - `POST /api/bulk-doc-analysis/export-profiles`
   - `PUT /api/bulk-doc-analysis/export-profiles/<id>`
   - `DELETE /api/bulk-doc-analysis/export-profiles/<id>`

5. Update download endpoints:
   - Generate export format based on workflow
   - Support all formats

**Deliverable:** Can export in CSV, JSON, MD, DOCX, PDF

---

### Phase 6: Admin UI - Workflow Builder (Week 15-16)

**Goal:** Admin panel for workflow creation

**Tasks:**
1. Create admin route: `/bulk-doc-analysis/admin/workflows`

2. Workflow metadata form:
   - Name (3-80 chars)
   - Description (20-240 chars, required)
   - Domain(s) multi-select

3. Ingestion & export setup:
   - Upload types selector
   - Ingestion mode (Programmatic / Vision)
   - Vision prompt textarea (if vision mode)
   - Export type selector

4. Prompt chain editor:
   - Existing chain editor
   - Add `title` field to each step (required)
   - Step titles shown in Runner UI

5. Workflow management:
   - List workflows
   - Edit (creates version)
   - Delete
   - View versions

**Deliverable:** Admins can create/edit workflows via UI

---

### Phase 7: Runner UI Redesign (Week 17-18)

**Goal:** Workflow execution UI

**Tasks:**
1. Redesign `/bulk-doc-analysis` route:
   - Remove chain selection
   - Add workflow selection (cards)

2. Workflow cards:
   - Name, domains, description
   - Step titles (first 3, then "+N more")
   - Input types, ingestion mode, export type
   - Step count, last updated

3. Search & filtering:
   - Search by name/description
   - Filter by domain
   - Filter by input type

4. Upload section:
   - File picker filtered by workflow
   - Show ingestion mode info
   - CSV row count preview

5. Execution progress:
   - Per-document (non-CSV)
   - Per-row/task (CSV)

6. Download:
   - Format matches workflow export
   - CSV workflows: compiled CSV

**Deliverable:** Users can run workflows via clean UI

---

### Phase 8: Step Titles & Polish (Week 19)

**Goal:** UX enhancements

**Tasks:**
1. Add `title` field to chain steps (database + UI)
2. Update chain creation/editing to require titles
3. Display step titles in Runner UI
4. Workflow card polish
5. Token counting enhancements (ingestion-time estimation)

**Deliverable:** Polished UX with step titles

---

## File Structure (What We're Modifying)

### New Files
```
external/ai_bulk_doc_analysis/
├── workflow_service.py          # NEW
├── ingestion_service.py         # NEW
├── export_service.py            # NEW
├── domain_service.py            # NEW
└── workers/
    └── export_worker.py         # NEW
```

### Modified Files
```
external/ai_bulk_doc_analysis/
├── models.py                    # Add new models
├── db_schema.sql                # Add new tables
├── db_service.py                # Add workflow methods
├── services.py                  # Enhance for multi-file
├── blueprint.py                 # Remove old endpoints, add new
├── workers/
│   ├── conversion_worker.py     # Multi-format + vision
│   └── execution_worker.py      # CSV task support
└── templates/
    └── bulk_doc_analysis.html    # Redesign Runner UI

web/
└── app.py                       # Register new admin routes

config/
└── users.json                   # Add domains field
```

### Files We're NOT Touching
```
external/agent/                  # ❌ DO NOT MODIFY
external/tools/parquet_agent/    # ❌ DO NOT MODIFY
tests/agent/                     # ❌ DO NOT MODIFY
```

---

## Dependencies to Add

```txt
# Ingestion
python-docx>=1.1.0
PyMuPDF>=1.23.0  # Alternative PDF library
pandas>=2.0.0

# Vision
Pillow>=10.0.0
pdf2image>=1.16.0

# Export
reportlab>=4.0.0
```

---

## Testing Strategy

### Unit Tests
- Workflow CRUD operations
- Domain access control
- Ingestion (programmatic + vision)
- Export generation
- CSV task creation

### Integration Tests
- End-to-end workflow execution
- CSV workflow (upload → tasks → execution → export)
- Vision ingestion (PDF → images → Claude → R0)
- Domain filtering

### Manual Testing
- Admin workflow builder UI
- Runner UI workflow cards
- Multi-file type upload
- Export formats

---

## Migration Notes

### Data Migration
- **No migration needed** - We're removing old endpoints
- Existing runs can be left as-is (or deleted)
- Fresh start with workflow system

### Backward Compatibility
- **None** - Clean break
- Old chain endpoints removed
- Users must use new workflow system

---

## Success Criteria

✅ Workflows can be created/edited/deleted  
✅ Domain-based access control works  
✅ Multi-file types can be ingested  
✅ Vision ingestion works for PDF  
✅ CSV workflows execute per-row  
✅ All export formats work  
✅ Admin can build workflows via UI  
✅ Users can run workflows via Runner UI  
✅ Step titles displayed in UI  
✅ No impact on Parquet Agent  

---

## Ready to Start? ✅

**Status:** All decisions confirmed, scope clear, plan ready.

**Next Step:** Begin Phase 1 - Core Workflow System

**Estimated Timeline:** 19 weeks (can be parallelized)

**Waiting for:** Your go-ahead to start implementation

