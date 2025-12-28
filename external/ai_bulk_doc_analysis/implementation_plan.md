# Implementation Plan: Worker/Queue + DB + Features 3 & 5

## Phase 1: Database Migration

### 1.1 Create Schema
- Run `db_schema.sql` to create all tables
- Add Alembic/migrations for version control

### 1.2 Data Models (SQLAlchemy)
- `Session`, `Document`, `Chain`, `ChainStep`, `ChainVersion`, `Run`, `StepResult`
- Replace in-memory Dict/List storage with DB queries

### 1.3 Migration Script
- Export existing in-memory data to DB (if any)
- Validate data integrity

---

## Phase 2: Worker/Queue Architecture (RQ)

### 2.1 Setup RQ
```bash
pip install rq redis
```

### 2.2 Create Two Queues
- `conversion_queue` - PDF → Markdown conversion
- `execution_queue` - Claude step execution

### 2.3 Worker Processes
- `workers/conversion_worker.py` - Handles `CONVERT_DOC` jobs
- `workers/execution_worker.py` - Handles `EXECUTE_STEP` jobs

### 2.4 API Changes
- `POST /documents/upload` → Enqueue `CONVERT_DOC` (don't convert inline)
- `POST /runs` → Enqueue `EXECUTE_STEP` jobs (don't execute inline)
- Keep polling endpoints unchanged

### 2.5 Idempotency
- Use `(run_id, doc_id, step_index)` unique constraint
- Worker checks StepResult before executing
- Skip if already `SUCCESS`

---

## Phase 3: Bulk Download (Feature #3)

### 3.1 API Endpoint
```
GET /api/bulk-doc-analysis/runs/<run_id>/download-all
```

### 3.2 Implementation
- Query all documents in run
- Get final R(N) output for each
- Create ZIP archive
- Stream download

### 3.3 Frontend
- Add "Download All" button in Panel 3
- Show progress during ZIP creation

---

## Phase 4: Model Configuration (Feature #5)

### 4.1 Database
- `chain_steps.model_config` JSONB column (already in schema)
- Default: `{"model": "claude-3-haiku-20240307", "max_tokens": 4096, "temperature": 0.2}`

### 4.2 Chain Editor UI
- Add model selector per step (Haiku/Sonnet/Opus)
- Add `max_tokens` input (default 4096)
- Add `temperature` slider (0.0-1.0, default 0.2)

### 4.3 Backend
- Store `model_config` in `chain_steps` table
- Pass to worker in `EXECUTE_STEP` job
- Worker uses config when calling Claude

### 4.4 StepResult Tracking
- Store `model`, `max_tokens`, `temperature` in `step_results` table
- Display in run progress UI

---

## Implementation Order

1. **Week 1**: Database schema + migration (Phase 1)
2. **Week 2**: RQ setup + conversion worker (Phase 2.1-2.3)
3. **Week 3**: Execution worker + API changes (Phase 2.4-2.5)
4. **Week 4**: Bulk download (Phase 3)
5. **Week 5**: Model configuration (Phase 4)

---

## Testing Strategy

- Unit tests for workers (mock Claude API)
- Integration tests for queue → worker → DB flow
- End-to-end: Upload → Convert → Run → Download
- Load test: Multiple concurrent runs

