# Implementation Plan: Defer Ingestion & Fix Step Column

## Overview
Two changes:
1. **Defer ingestion to Step 3**: Documents uploaded in Step 1 remain QUEUED. Ingestion (vision/programmatic) happens when "Start Processing" is clicked in Step 3, before workflow steps execute.
2. **Fix Step column**: Show actual progress (R0 → R1 → R2...) instead of always showing final step number.

---

## Change 1: Defer Ingestion to Step 3

### Current Flow
```
Step 1: Upload → Immediate conversion → Status CONVERTED
Step 2: Select format
Step 3: Start Processing → Workflow steps (R1, R2, R3...)
```

### New Flow
```
Step 1: Upload → Status QUEUED (no conversion)
Step 2: Select format
Step 3: Start Processing → 
  1. Convert/ingest all docs (create R0.md) → Status CONVERTED
  2. Then execute workflow steps (R1, R2, R3...)
```

### Implementation Steps

#### 1.1 Remove Automatic Conversion on Upload
**File:** `external/ai_bulk_doc_analysis/blueprint.py`

**Location:** Lines 219-252 in `api_upload_documents()`

**Change:**
- Remove the code that enqueues conversion jobs immediately after upload
- Keep document creation, but leave status as "QUEUED" (default)
- Documents should remain in QUEUED status until Step 3

```python
# REMOVE THIS BLOCK (lines 219-252):
# Enqueue conversion jobs if queues are enabled
if USE_QUEUES:
    try:
        conversion_queue = get_conversion_queue()
        for doc in docs:
            # ... conversion job enqueueing ...
```

#### 1.2 Add Conversion to Run Creation
**File:** `external/ai_bulk_doc_analysis/blueprint.py`

**Location:** `api_create_run()` function (around line 1400-1540)

**Change:**
- Before creating workflow execution jobs, first convert all documents
- Enqueue conversion jobs for all QUEUED documents
- Wait for conversions to complete (or enqueue them with proper dependencies)
- Then proceed with workflow step execution

**Implementation:**
```python
# In api_create_run(), before workflow step enqueueing:

# Step 1: Convert all documents first
unconverted_docs = [d for d in db_docs if d.status != "CONVERTED"]
if unconverted_docs:
    # Get ingestion profile from workflow
    ingestion_profile_id = workflow_version.ingestion_profile_id
    
    # Enqueue conversion jobs
    if USE_QUEUES:
        conversion_queue = get_conversion_queue()
        for doc in unconverted_docs:
            # Find file path
            doc_dir = storage_base / "docs" / doc.doc_id
            file_path = None
            for ext in ['.pdf', '.docx', '.txt', '.md', '.csv']:
                for f in doc_dir.glob(f"*{ext}"):
                    file_path = f
                    break
                if file_path:
                    break
            
            if file_path:
                object_storage_key = str(file_path.relative_to(storage_base))
                job_data = {
                    "doc_id": doc.doc_id,
                    "session_id": session_id,
                    "object_storage_key": object_storage_key,
                    "ingestion_profile_id": ingestion_profile_id,
                    "idempotency_key": f"convert:{doc.doc_id}",
                }
                conversion_queue.enqueue(
                    convert_doc_job, 
                    job_data, 
                    job_id=f"convert_{doc.doc_id}"
                )
    
    # For synchronous execution (no queues), convert inline
    else:
        from external.ai_bulk_doc_analysis.ingestion_service import IngestionService
        ingestion_service = IngestionService()
        ingestion_profile = ingestion_service.get_ingestion_profile(ingestion_profile_id)
        
        for doc in unconverted_docs:
            # Find and convert file
            doc_dir = storage_base / "docs" / doc.doc_id
            file_path = None
            for ext in ['.pdf', '.docx', '.txt', '.md', '.csv']:
                for f in doc_dir.glob(f"*{ext}"):
                    file_path = f
                    break
                if file_path:
                    break
            
            if file_path:
                try:
                    r0_content, metadata = ingestion_service.ingest_file(file_path, ingestion_profile)
                    # Save R0.md
                    output_dir = storage_base / "sessions" / session_id / "docs" / doc.doc_id
                    output_dir.mkdir(parents=True, exist_ok=True)
                    r0_path = output_dir / "R0.md"
                    r0_path.write_text(r0_content, encoding='utf-8')
                    
                    # Update document status
                    with get_db_session() as db:
                        db_doc = db.query(Document).filter(Document.doc_id == doc.doc_id).first()
                        if db_doc:
                            db_doc.status = "CONVERTED"
                            db_doc.converted_md_path = str(r0_path)
                            db.commit()
                except Exception as e:
                    logger.error(f"Failed to convert doc {doc.doc_id}: {e}")
                    # Mark as ERROR or continue?

# Step 2: Now proceed with workflow step execution (existing code)
```

#### 1.3 Update UI Validation
**File:** `external/ai_bulk_doc_analysis/static/bulk_doc_analysis.js`

**Location:** `updateStartButton()` and `startProcessing()`

**Change:**
- Remove requirement that all docs must be CONVERTED before Step 3
- Allow Step 3 to proceed with QUEUED documents
- The conversion will happen when "Start Processing" is clicked

```javascript
// In updateStartButton():
// REMOVE: const allConverted = state.docs.every((d) => d.status === Status.CONVERTED);
// CHANGE: const canStart = hasWorkflow && hasDocs && hasFormat; // Remove allConverted check

// In startProcessing():
// REMOVE: const allConverted = state.docs.every(d => d.status === Status.CONVERTED);
// REMOVE: if (!allConverted) { toast('error', ...); return; }
```

#### 1.4 Update Document Status Polling
**File:** `external/ai_bulk_doc_analysis/static/bulk_doc_analysis.js`

**Change:**
- Keep polling active during Step 3 to show conversion progress
- Don't auto-advance to Step 2 based on conversion status
- Show conversion status in Step 3 table

---

## Change 2: Fix Step Column to Show Progress

### Current Behavior
- Shows "R3" immediately for 3-step workflow (wrong)
- Doesn't update as steps complete

### New Behavior
- Shows "R0" initially (after conversion)
- Updates to "R1" when step 1 completes
- Updates to "R2" when step 2 completes
- Updates to "R3" when step 3 completes

### Implementation

#### 2.1 Fix Step Label Calculation
**File:** `external/ai_bulk_doc_analysis/db_service.py`

**Location:** `get_run_progress()` function, lines 915-916 and 838-839

**Change:**
Only count steps that have completed successfully (status == "SUCCESS")

**For Non-CSV workflows (line 915):**
```python
# OLD:
max_step = max([sr.step_index for sr in doc_step_results], default=0)

# NEW:
completed_steps = [sr for sr in doc_step_results if sr.status == "SUCCESS"]
max_step = max([sr.step_index for sr in completed_steps], default=0)
```

**For CSV workflows (line 838):**
```python
# OLD:
max_step = max([sr.step_index for sr in task_results], default=0)

# NEW:
completed_steps = [sr for sr in task_results if sr.status == "SUCCESS"]
max_step = max([sr.step_index for sr in completed_steps], default=0)
```

**Alternative (show current step including RUNNING):**
```python
# Show highest step that's either SUCCESS or RUNNING
active_steps = [sr for sr in doc_step_results if sr.status in ["SUCCESS", "RUNNING"]]
max_step = max([sr.step_index for sr in active_steps], default=0)
```

#### 2.2 Handle R0 Display
**File:** `external/ai_bulk_doc_analysis/db_service.py`

**Change:**
- When no workflow steps have completed, show "R0" (converted document)
- R0 is not a StepResult, it's the converted document
- Logic: If max_step == 0, show "R0". Otherwise show "R{max_step}"

```python
# After calculating max_step:
if max_step == 0:
    # Check if document is converted (has R0.md)
    if db_doc.status == "CONVERTED" or db_doc.converted_md_path:
        step_label = "R0"
    else:
        step_label = "—"  # Not converted yet
else:
    step_label = f"R{max_step}"
```

---

## Testing Checklist

### Change 1: Defer Ingestion
- [ ] Upload documents in Step 1 → Status should be QUEUED
- [ ] Step 3 button should be enabled without waiting for conversion
- [ ] Click "Start Processing" → Documents should convert first
- [ ] After conversion, workflow steps should execute
- [ ] Vision ingestion should work correctly when triggered in Step 3
- [ ] Programmatic ingestion should work correctly when triggered in Step 3

### Change 2: Fix Step Column
- [ ] After conversion, Step column shows "R0"
- [ ] After step 1 completes, Step column shows "R1"
- [ ] After step 2 completes, Step column updates to "R2"
- [ ] After step 3 completes, Step column updates to "R3"
- [ ] CSV workflows show correct step per task
- [ ] Step column updates in real-time during polling

---

## Step Column Update Mechanism (Real-time Updates)

### ✅ Frontend Already Supports Instant Updates

**Confirmed Implementation:**
1. **Polling mechanism**: `startPollingRunProgress()` polls every 2 seconds (line 2501 in bulk_doc_analysis.js)
2. **Backend fetch**: Fetches `step_label` from `/api/bulk-doc-analysis/runs/{runId}/progress` (line 2504)
3. **State update**: Updates `state.run.rows` with new `stepLabel` from backend (line 2512)
4. **UI re-render**: Calls `renderRun()` which re-renders table with updated `stepLabel` (line 2519, displays at line 387)

**Code flow:**
```javascript
// Polling every 2 seconds (line 2501)
setInterval(async () => {
  const progress = await api.getRunProgress(runId);
  state.run.rows = progress.rows.map((r) => ({
    stepLabel: r.step_label || 'R0',  // Reads from backend
    // ... other fields
  }));
  renderRun();  // Re-renders table with new stepLabel
}, 2000);
```

**Conclusion:**
- ✅ **No frontend work needed** - Update mechanism already exists
- ✅ Step column **WILL update automatically** once backend calculation is fixed
- ✅ Updates happen **every 2 seconds** (existing polling interval)
- ✅ **No code changes required** in frontend

**Note:** The polling starts automatically when "Start Processing" is clicked (line 2311).

---

## Files to Modify

1. `external/ai_bulk_doc_analysis/blueprint.py`
   - Remove conversion enqueueing from `api_upload_documents()`
   - Add conversion logic to `api_create_run()`

2. `external/ai_bulk_doc_analysis/db_service.py`
   - Fix `get_run_progress()` step_label calculation

3. `external/ai_bulk_doc_analysis/static/bulk_doc_analysis.js`
   - Update `updateStartButton()` validation
   - Update `startProcessing()` validation
   - Adjust polling logic

---

## Edge Cases

1. **Partial conversion failure**: If some docs fail to convert, should workflow still proceed?
   - Recommendation: Only proceed with successfully converted docs

2. **Conversion timeout**: If conversion takes too long, should workflow wait?
   - Recommendation: Use job dependencies or wait for conversion completion

3. **No workflow steps**: If workflow has 0 steps, what to show?
   - Recommendation: Show "R0" (converted document is the final output)

4. **Step failures**: If step 2 fails, should step 3 still show?
   - Recommendation: Show highest successfully completed step

