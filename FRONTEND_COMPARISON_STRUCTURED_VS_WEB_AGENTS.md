# Frontend Comparison: Structured vs Web Research Agents

## Summary

**Status**: ⚠️ **Web Research Agent Frontend is Partially Complete**

- ✅ Admin UI: Complete for both agent types
- ✅ Chat Interface: Generic, works for both
- ⚠️ **Missing**: Web research-specific UI components (evidence pack display, sources, claims, conflicts)

---

## Admin Panel Comparison

### ✅ Structured Agents Admin UI

**Location**: `/admin` → "Manage Agents" → "Structured"

**Features**:
- ✅ List agents (`agent_type="structured"`)
- ✅ Create new agent form
- ✅ Edit agent form
- ✅ Delete agent
- ✅ Fields:
  - Agent Name
  - Description
  - LLM Model (dropdown)
  - Domain File (upload)
  - Data Folder (select existing or create new)
  - Data Files (upload CSV/Parquet)

**Code**: `web/templates/admin.html` lines 609-696

### ✅ Web Research Agents Admin UI

**Location**: `/admin` → "Manage Agents" → "External"

**Features**:
- ✅ List agents (`agent_type="external"`)
- ✅ Create new agent form
- ✅ Edit agent form
- ✅ Delete agent
- ✅ Fields:
  - Agent Name
  - Description
  - LLM Model (dropdown)
  - Domain File (upload)
  - **Tavily API Key** (password field)
  - **Allowed Domains** (comma-separated)
  - **Blocked Domains** (comma-separated)
  - **Default Research Depth** (quick/standard/deep)

**Code**: `web/templates/admin.html` lines 709-803

**Status**: ✅ **Complete** - All web research-specific fields are present

---

## System Prompts Management

### ✅ Structured Agents Prompts

**Location**: `/admin` → "System Prompts" → "Structured"

**Prompts Managed**:
- `decider.md`
- `nl_to_sql_planner.md`
- `sql_plan_updater.md`
- `query_result_evaluator.md`
- `ask_user_clarification.md`
- `response_commentary.md`

**Code**: `web/templates/admin.html` lines 577-586

### ✅ Web Research Agents Prompts

**Location**: `/admin` → "System Prompts" → "External"

**Prompts Managed**:
- `web_research_decider.md`
- `web_research_synthesis.md`
- `web_research_claim_extraction.md`
- `web_research_conflict_detection.md`
- `web_research_ask_user_clarification.md`
- `web_research_response_commentary.md`

**Code**: `web/templates/admin.html` lines 599-607

**Status**: ✅ **Complete** - All web research prompts are manageable via UI

---

## Chat Interface Comparison

### ✅ Generic Chat Interface

**Location**: `/agent/chat/<agent_id>`

**Template**: `web/templates/agent_chat.html`

**Features** (Works for both agent types):
- ✅ Message history display
- ✅ User input field
- ✅ Send button
- ✅ Conversation sidebar
- ✅ New chat button
- ✅ Show thinking toggle
- ✅ SSE streaming support
- ✅ Markdown rendering
- ✅ Error handling
- ✅ Loading indicators

**Status**: ✅ **Complete** - Generic chat interface works for both types

---

## Response Display Comparison

### ✅ Structured Agent Response Display

**Features**:
- ✅ Markdown rendering for text responses
- ✅ **SQL Query display** (code block with syntax highlighting)
- ✅ **SQL Results table** (DataTables with search, sort, pagination)
- ✅ Row count display
- ✅ Export buttons (copy, CSV)
- ✅ Thinking panel (collapsible)

**Code**: `web/templates/agent_chat.html`
- `displaySQLResults()` - lines 770-811
- `formatTable()` - lines 815-887
- SQL display - lines 789-801

**SSE Events Handled**:
- `sql_generated` - Updates UI with SQL
- `results_ready` - Shows row count
- `final_response` - Displays markdown response + table

### ⚠️ Web Research Agent Response Display

**Current Implementation**:
- ✅ Markdown rendering for `final_answer`
- ✅ Generic message display
- ❌ **Missing**: Evidence pack display
- ❌ **Missing**: Sources list with URLs
- ❌ **Missing**: Claims display
- ❌ **Missing**: Conflicts display
- ❌ **Missing**: Gaps display
- ❌ **Missing**: Citations/source attribution

**Backend Data Available** (from `agent_run_routes.py` lines 627-657):
```python
final_answer = result.get("final_answer")
evidence_pack = result.get("evidence_pack", {})
sources = evidence_pack.get("sources", [])  # List of {url, title, snippet}
claims = evidence_pack.get("claims", [])    # List of extracted claims
conflicts = evidence_pack.get("conflicts", [])  # List of conflicts
```

**SSE Events**:
- `final_response` - Emits `{"response": final_answer, "evidence_pack": evidence_pack}`
- But frontend doesn't parse/display `evidence_pack`

**Status**: ⚠️ **Incomplete** - Response text displays, but evidence pack data is ignored

---

## Missing Frontend Components for Web Research Agents

### 1. Evidence Pack Display Component

**What's Needed**:
```javascript
function displayEvidencePack(messageEl, evidencePack) {
    // Display sources, claims, conflicts, gaps
    // Similar to displaySQLResults() but for research data
}
```

**Should Display**:
- **Sources Section**: List of URLs with titles, snippets, clickable links
- **Claims Section**: Extracted claims with confidence levels
- **Conflicts Section**: Conflicting claims with severity indicators
- **Gaps Section**: Information gaps identified

### 2. Sources List Component

**What's Needed**:
- Collapsible section showing all sources
- Each source: URL (clickable), title, snippet, date
- Group by domain/authority
- Search/filter sources

### 3. Claims Display Component

**What's Needed**:
- List of extracted claims
- Confidence indicators (high/medium/low)
- Source attribution for each claim
- Expandable details

### 4. Conflicts Display Component

**What's Needed**:
- Highlight conflicting claims
- Show severity (high/medium/low)
- Show which sources disagree
- Visual indicators (color coding)

### 5. Citations Component

**What's Needed**:
- Inline citations in the final answer
- Clickable citation numbers/links
- Source attribution per paragraph/claim

---

## Backend Support Status

### ✅ Structured Agent Backend

**Events Emitted**:
- `run_started`
- `decider_done`
- `sql_generated` (with SQL text)
- `results_ready` (with row count)
- `final_response` (with response text)
- `run_completed`

**Data Structure**:
```json
{
  "status": "SUCCESS",
  "final_sql": "SELECT ...",
  "finished_output": {
    "response": "...",
    "sql": "SELECT ...",
    "results": {
      "columns": [...],
      "rows": [...],
      "row_count": 100
    }
  }
}
```

### ✅ Web Research Agent Backend

**Events Emitted**:
- `run_started`
- `decider_done`
- `final_response` (with `response` and `evidence_pack`)
- `run_completed`

**Data Structure**:
```json
{
  "status": "SUCCESS",
  "final_answer": "...",
  "evidence_pack": {
    "sources": [
      {"url": "...", "title": "...", "snippet": "..."}
    ],
    "claims": [
      {"text": "...", "confidence": "high", "source": "..."}
    ],
    "conflicts": [
      {"claim1": "...", "claim2": "...", "severity": "high"}
    ],
    "gaps": [
      {"description": "...", "criticality": "high"}
    ]
  }
}
```

**Status**: ✅ **Complete** - Backend provides all necessary data

---

## Frontend Code Gaps

### Missing Functions

1. **`displayEvidencePack(messageEl, evidencePack)`**
   - Should be similar to `displaySQLResults()`
   - Create collapsible sections for sources, claims, conflicts, gaps

2. **`formatSourcesList(sources)`**
   - Format sources as a list with links
   - Group by domain
   - Add search/filter

3. **`formatClaimsList(claims)`**
   - Display claims with confidence indicators
   - Show source attribution

4. **`formatConflictsList(conflicts)`**
   - Highlight conflicts
   - Show severity
   - Visual indicators

### Missing SSE Event Handlers

**Current** (lines 1328-1458):
- Handles `sql_generated`, `results_ready` (structured-specific)
- Handles `final_response` but only displays text, ignores `evidence_pack`

**Needed**:
- Parse `evidence_pack` from `final_response` event
- Call `displayEvidencePack()` when evidence_pack is present
- Handle web research-specific events (if any)

### Missing UI Elements

1. **Evidence Pack Section** (collapsible)
   - Sources tab
   - Claims tab
   - Conflicts tab
   - Gaps tab

2. **Source Cards**
   - URL (clickable)
   - Title
   - Snippet
   - Domain badge
   - Date (if available)

3. **Claim Cards**
   - Claim text
   - Confidence badge
   - Source link
   - Expandable details

4. **Conflict Indicators**
   - Visual conflict markers
   - Severity badges
   - Source comparison

---

## Implementation Recommendations

### Priority 1: Basic Evidence Pack Display

Add to `agent_chat.html`:

```javascript
function displayEvidencePack(messageEl, evidencePack) {
    if (!evidencePack) return;
    
    const evidenceDiv = document.createElement('details');
    evidenceDiv.className = 'evidence-pack mt-3';
    evidenceDiv.open = true;
    evidenceDiv.innerHTML = `
        <summary class="small text-info" style="cursor:pointer;">
            <i class="bi bi-file-earmark-text"></i> Research Evidence 
            (${evidencePack.sources?.length || 0} sources, 
             ${evidencePack.claims?.length || 0} claims)
        </summary>
        <div class="evidence-content p-3 bg-light border rounded mt-2">
            ${formatSourcesList(evidencePack.sources || [])}
            ${formatClaimsList(evidencePack.claims || [])}
            ${formatConflictsList(evidencePack.conflicts || [])}
        </div>
    `;
    messageEl.appendChild(evidenceDiv);
}
```

### Priority 2: Update SSE Handler

Modify `final_response` handler (line ~1360):

```javascript
case 'final_response':
    const response = payload.response || '';
    const evidencePack = payload.evidence_pack; // ADD THIS
    
    if (markdownBody && typeof marked !== 'undefined') {
        markdownBody.innerHTML = marked.parse(response);
    }
    
    // ADD: Display evidence pack if present
    if (evidencePack && messageEl) {
        displayEvidencePack(messageEl, evidencePack);
    }
    break;
```

### Priority 3: Format Functions

Add formatting functions similar to `formatTable()`:

```javascript
function formatSourcesList(sources) {
    // Return HTML for sources list
}

function formatClaimsList(claims) {
    // Return HTML for claims list
}

function formatConflictsList(conflicts) {
    // Return HTML for conflicts list
}
```

---

## Comparison Table

| Feature | Structured Agent | Web Research Agent | Status |
|---------|-----------------|-------------------|--------|
| **Admin UI - List Agents** | ✅ | ✅ | Complete |
| **Admin UI - Create Agent** | ✅ | ✅ | Complete |
| **Admin UI - Edit Agent** | ✅ | ✅ | Complete |
| **Admin UI - Delete Agent** | ✅ | ✅ | Complete |
| **Admin UI - Agent Fields** | ✅ (data_folder, data_files) | ✅ (tavily_api_key, domains, research_depth) | Complete |
| **System Prompts Management** | ✅ | ✅ | Complete |
| **Chat Interface** | ✅ | ✅ | Complete |
| **Message Display** | ✅ | ✅ | Complete |
| **Markdown Rendering** | ✅ | ✅ | Complete |
| **SQL Query Display** | ✅ | N/A | N/A |
| **SQL Results Table** | ✅ | N/A | N/A |
| **Evidence Pack Display** | N/A | ❌ | **Missing** |
| **Sources List** | N/A | ❌ | **Missing** |
| **Claims Display** | N/A | ❌ | **Missing** |
| **Conflicts Display** | N/A | ❌ | **Missing** |
| **Citations** | N/A | ❌ | **Missing** |
| **SSE Event Handling** | ✅ (sql_generated, results_ready) | ⚠️ (final_response partial) | Partial |

---

## Conclusion

**What's Complete**:
- ✅ Admin UI for both agent types
- ✅ System prompts management
- ✅ Generic chat interface
- ✅ Basic message display
- ✅ Backend data provision

**What's Missing**:
- ❌ Evidence pack display component
- ❌ Sources list component
- ❌ Claims display component
- ❌ Conflicts display component
- ❌ Citations/inline source attribution
- ❌ SSE event handler for evidence_pack

**Recommendation**: 
Implement the missing UI components to fully support web research agent responses. The backend already provides all necessary data (`evidence_pack`), but the frontend doesn't parse or display it.

