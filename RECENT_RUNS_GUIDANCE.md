# Recent Runs Feature - Implementation Guidance

## Overview
Display previous workflow runs in the "Recent Runs" section so users can:
- View past runs with workflow template name, date/time, status
- Reload/view a workflow run from a previous session
- See runs from previous days and navigate back to them

## Current State
- **Location:** `external/ai_bulk_doc_analysis/templates/bulk_doc_analysis.html` - "Recent Runs" section (currently shows "Recent runs will appear here")
- **API Endpoint:** Already exists: `/api/bulk-doc-analysis/runs/<run_id>/progress`
- **Database:** Runs are stored in `bulk_doc_runs` table with `created_at`, `status`, `workflow_version_id`

## What Needs to Be Done

### 1. Backend API Changes

**New Endpoint:** `GET /api/bulk-doc-analysis/runs` (list all runs for current user)

**Response Format:**
```json
{
  "runs": [
    {
      "run_id": "uuid",
      "workflow_name": "Task Analysis & Recommendations",
      "workflow_version_id": "wfv_...",
      "status": "COMPLETE",
      "created_at": "2026-01-03T13:51:36Z",
      "completed_at": "2026-01-03T13:54:09Z",
      "total_input_tokens": 5679,
      "total_output_tokens": 9441,
      "document_count": 1,  // or task_count for CSV
      "is_csv_workflow": false
    }
  ]
}
```

**Implementation Steps:**
1. Add route in `external/ai_bulk_doc_analysis/blueprint.py`
2. Query `bulk_doc_runs` table filtered by user's session(s)
3. Join with `workflow_versions` and `workflows` to get workflow name
4. Join with `execution_tasks` or count step_results to get document/task count
5. Order by `created_at DESC`
6. Limit to last 20-50 runs

### 2. Frontend Changes

**File:** `external/ai_bulk_doc_analysis/static/bulk_doc_analysis.js`

**New Functions Needed:**
```javascript
async function loadRecentRuns() {
  // Fetch runs from API
  // Update state.recentRuns
  // Call renderRecentRuns()
}

function renderRecentRuns() {
  // Create table rows for each run
  // Show: Workflow name, Date/Time, Status, Actions button
}

function openRunDetails(runId) {
  // Load run progress
  // Show in the run progress table (STEP 4)
  // Start polling if not complete
}
```

**UI Structure:**
```html
<table class="table">
  <thead>
    <tr>
      <th>Workflow Template</th>
      <th>Date & Time</th>
      <th>Status</th>
      <th>Documents/Tasks</th>
      <th>Actions</th>
    </tr>
  </thead>
  <tbody id="recentRunsTbody">
    <!-- Rows populated by renderRecentRuns() -->
  </tbody>
</table>
```

**Table Row Example:**
- Workflow Template: Link or bold text showing workflow name
- Date & Time: Formatted date/time (e.g., "Jan 3, 2026 1:51 PM")
- Status: Badge (SUCCESS, ERROR, RUNNING, QUEUED)
- Documents/Tasks: Count (e.g., "6 tasks" for CSV, "1 document" for PDF)
- Actions: Button "View" or "Open" that calls `openRunDetails(runId)`

### 3. Data Flow

1. **On Page Load:**
   - Call `loadRecentRuns()` when bulk doc analysis page loads
   - Display in "Recent Runs" section

2. **When User Clicks "View" on a Run:**
   - Call `openRunDetails(runId)`
   - Set `state.run.runId = runId`
   - Call `refreshRunProgress()` to load run data
   - Show run progress table (STEP 4)
   - Start polling if status is not COMPLETE/ERROR

3. **Session Handling:**
   - Option 1: Show runs for current user (all sessions)
   - Option 2: Show runs for current session only
   - **Recommendation:** Show all runs for user (more useful)

### 4. Database Queries Needed

```python
# In blueprint.py or db_service.py
def list_runs(user_id: str, limit: int = 50) -> List[Dict]:
    with get_db_session() as db:
        # Get all sessions for user
        sessions = db.query(DBSession).filter(
            DBSession.user_id == user_id
        ).all()
        session_ids = [s.session_id for s in sessions]
        
        # Get runs for those sessions
        runs = db.query(DBRun).filter(
            DBRun.session_id.in_(session_ids)
        ).order_by(DBRun.created_at.desc()).limit(limit).all()
        
        # Join with workflow to get name
        results = []
        for run in runs:
            workflow_name = "Unknown"
            if run.workflow_version_id:
                wfv = db.query(WorkflowVersion).filter(
                    WorkflowVersion.workflow_version_id == run.workflow_version_id
                ).first()
                if wfv:
                    workflow = db.query(Workflow).filter(
                        Workflow.workflow_id == wfv.workflow_id
                    ).first()
                    if workflow:
                        workflow_name = workflow.name
            
            # Count documents/tasks
            task_count = db.query(ExecutionTask).filter(
                ExecutionTask.run_id == run.run_id
            ).count()
            is_csv = task_count > 0
            
            results.append({
                "run_id": run.run_id,
                "workflow_name": workflow_name,
                "workflow_version_id": run.workflow_version_id,
                "status": run.status,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "total_input_tokens": run.total_input_tokens or 0,
                "total_output_tokens": run.total_output_tokens or 0,
                "document_count": 1 if not is_csv else task_count,
                "task_count": task_count if is_csv else 0,
                "is_csv_workflow": is_csv,
            })
        
        return results
```

### 5. UI/UX Considerations

- **Table Design:** Clean, sortable columns
- **Status Badges:** Color-coded (green=SUCCESS, red=ERROR, blue=RUNNING, gray=QUEUED)
- **Date Format:** User-friendly (e.g., "Today 1:51 PM", "Yesterday 2:30 PM", "Jan 2, 2026")
- **Pagination:** If many runs, add pagination (show 20 per page)
- **Empty State:** "No previous runs" message when list is empty
- **Loading State:** Show spinner while loading

### 6. Integration Points

- **Run Progress Table:** Reuse existing `refreshRunProgress()` function
- **Polling:** Reuse existing `startRunPolling()` when opening incomplete runs
- **Download:** Reuse existing download functionality (already works with run_id)

## Example Implementation Flow

1. User opens bulk doc analysis page
2. `loadRecentRuns()` fetches last 20 runs
3. `renderRecentRuns()` displays them in table
4. User clicks "View" on a run from yesterday
5. `openRunDetails(runId)` is called
6. Run progress is loaded and displayed in STEP 4 section
7. If run is COMPLETE, show all outputs
8. If run is QUEUED/RUNNING, start polling for updates
9. User can download outputs, view details, etc.

## Benefits

- **Persistent Access:** Users can return to previous workflows
- **Audit Trail:** See history of what was processed
- **Resume Work:** Continue working on incomplete runs
- **Better UX:** No need to remember run IDs or manually navigate

