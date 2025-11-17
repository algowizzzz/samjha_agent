# **Document Review Agent – Complete Implementation & Testing Specification**

**Version:** 3.0  
**Date:** November 15, 2025  
**Purpose:** End-to-end specification for building, testing, and validating a production-grade document review agent with autonomous agent layer and VS Code Web front-end.

---

## **Table of Contents**

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Agent State Schema](#3-agent-state-schema)
4. [Phase 0: Document Ingestion](#4-phase-0-document-ingestion)
5. [Phase 1: Holistic Assessment](#5-phase-1-holistic-assessment)
6. [Phase 2: Section-Level Deep Review](#6-phase-2-section-level-deep-review)
7. [Phase 3: Change Selection & Application](#7-phase-3-change-selection--application)
8. [Orchestrator & Control Flow](#8-orchestrator--control-flow)
9. [WebSocket Event Streaming](#9-websocket-event-streaming)
10. [UI/UX Design](#10-uiux-design)
11. [Test Strategy & Coverage](#11-test-strategy--coverage)
12. [Business Validation Criteria](#12-business-validation-criteria)
13. [Sample Documents & Expected Outcomes](#13-sample-documents--expected-outcomes)
14. [Implementation Checklist](#14-implementation-checklist)

---

## **1. Executive Summary**

### **1.1 What This Agent Does**

The Document Review Agent automates policy/procedure document quality assessment by:
1. **Ingesting** documents (PDF, DOCX, MD) and extracting structure
2. **Analyzing** content against a policy template (9-section standard)
3. **Identifying** gaps, tone issues, and structural problems
4. **Recommending** specific, line-level changes
5. **Applying** user-selected changes deterministically

### **1.2 Key Design Principles**

- **Configurability:** All LLM prompts stored as external `.md` files
- **Observability:** Every node/tool emits WebSocket events
- **Determinism:** Change application uses exact text replacement (no LLM rewriting)
- **Business-Friendly:** Outputs use plain language, not technical jargon
- **Testability:** Each phase has unit, integration, and E2E tests

### **1.3 User Journey (Natural Language Commands)**

The agent behaves like **"Cursor for documents"** – users interact via natural language:

1. **"Run full review on this document"** → Agent plans and executes Phase 0 → Phase 1 → Phase 2 → Phase 3
2. **"Re-run Phase 2 for Escalations only"** → Agent re-processes only the Escalations section
3. **"Apply all high severity changes"** → Agent selects and applies changes, shows diff in VS Code
4. **"Show me the template fitness report"** → Agent opens `/phase1/template_fitness.json` in editor

All artifacts (summaries, reviews, changes, diffs) are visible in a **VS Code Web interface** with:
- **Left:** File explorer (virtual file system)
- **Middle:** Editor/diff viewer
- **Right:** Agent console (chat + execution trace)

---

## **2. System Architecture**

### **2.1 Component Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                  Autonomous Document Review Agent                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │         Agent Layer (Natural Language Interface)       │    │
│  │  - agent_planner_llm (interprets user commands)        │    │
│  │  - Fixed tool set: run_phase1/2/3, apply_changes, etc │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Core Orchestrator Layer                   │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │    │
│  │  │MCP Tools │   │LLM Nodes │   │Control Flow Logic│  │    │
│  │  │(6 tools) │   │(10 nodes)│   │(state machine)   │  │    │
│  │  └──────────┘   └──────────┘   └──────────────────┘  │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              AgentState (TypedDict)                     │    │
│  │  - doc_meta, structure, template_meta                  │    │
│  │  - phase1, phase2, changes                             │    │
│  │  - user_interaction, control_state                     │    │
│  │  - vfs_artifacts (file paths for VS Code)              │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │      WebSocket Event Stream                   │
        │  - node_started / node_completed              │
        │  - vfs_file_updated                           │
        │  - open_artifact                              │
        └───────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │         VS Code Web Front-End                 │
        │  ┌─────────────┐ ┌──────────┐ ┌───────────┐  │
        │  │  Explorer   │ │  Editor  │ │  Agent    │  │
        │  │  (VFS tree) │ │  (Diff)  │ │  Console  │  │
        │  │  Left       │ │  Middle  │ │  Right    │  │
        │  └─────────────┘ └──────────┘ └───────────┘  │
        └───────────────────────────────────────────────┘
```

### **2.2 Technology Stack**

- **Backend:** Python 3.11+, LangGraph (orchestration)
- **LLM:** Claude 3 Opus (via Anthropic API)
- **MCP Tools:** Custom Python implementations
- **WebSocket:** Socket.IO
- **Storage:** JSON files (state snapshots)
- **Front-End:** VS Code Web (OSS build), custom VFS provider, WebView for Agent Console
- **VFS Backend:** REST API (`/vfs/file`, `/vfs/tree`) for virtual file system

---

### **2.3 Agent Layer (Autonomous Orchestrator)**

#### **2.3.1 Overview**

On top of the MCP tools, LLM nodes, and orchestrator, we introduce a **DocReviewAgent** that:

* Accepts **natural language commands** from the user (via chat)
* Plans **next steps** (which phases/nodes/tools to run)
* Invokes the underlying orchestrator with the right `control` values
* Updates the virtual file system (VFS) so the front-end (VS Code) reflects all artifacts and diffs

The agent behaves like a **"Cursor for documents"**:

* User can:
  * Run full workflow: **"Run full review on this document"**
  * Run/re-run phases: **"Re-run Phase 1 with the new template"**, **"Re-run Phase 2 for Escalations only"**
  * Apply changes: **"Apply all high severity issues"**, **"Apply 1,2,3,4 only"**
  * Navigate artifacts: **"Show me the template fitness report"**

#### **2.3.2 Agent Commands / Capabilities**

The agent supports a fixed set of "tools" it can call:

* `run_phase1(doc_id)` – Execute Phase 0 + Phase 1 (ingestion + holistic assessment)
* `run_phase2(doc_id, section_scope=None)` – Execute Phase 2 (section extraction + review)
  * `section_scope` = all sections | list of sections (e.g., `["Escalations", "Governance"]`)
* `run_phase3(doc_id)` – Execute Phase 3 (change application)
* `rerun_phase1(doc_id)` – Re-run Phase 1 with updated template or config
* `rerun_phase2(doc_id, section_scope)` – Re-run Phase 2 for specific sections
* `rerun_section(doc_id, section_title)` – Re-run review for a single section
* `apply_changes(doc_id, change_ids=None, severity_filter=None, sections=None)` – Apply selected changes
  * `change_ids`: List of specific change IDs (e.g., `["CHG-001", "CHG-002"]`)
  * `severity_filter`: `"high"` | `"medium"` | `"low"`
  * `sections`: List of section names (e.g., `["Governance", "Escalations"]`)
* `open_artifact(path)` – Open a file in the VS Code editor (e.g., `/phase1/doc_summary.md`)

#### **2.3.3 Agent Planning Node (LLM)**

**New LLM Node:** `agent_planner_llm`

**Prompt File:** `external/doc_review/prompts/agent_planner.md`

**Purpose:** Interpret user's natural language command and generate a plan of tool calls.

**Inputs:**
```python
{
    "user_message": str,                 # "Run full review on this document"
    "state_summary": {                   # High-level state snapshot
        "doc_id": str,
        "phase1_done": bool,
        "phase2_done": bool,
        "phase3_done": bool,
        "total_changes": int,
        "high_severity_count": int
    },
    "available_tools": List[str]         # Tool names from 2.3.2
}
```

**Prompt Behavior:**
- Must choose from the **fixed tool set** only (no arbitrary code)
- Must keep plan **short and explicit** (2–5 steps)
- Does **not** edit doc text directly; only orchestrates phases + apply tools
- Provides clear explanation for the plan

**Outputs:**
```python
{
    "plan_steps": [
        {"action": "run_phase1", "args": {"doc_id": "DOC-123"}},
        {"action": "run_phase2", "args": {"doc_id": "DOC-123"}},
        {"action": "open_artifact", "args": {"path": "/phase2/summary_report.json"}}
    ],
    "explanation": "User asked to run a full review. Will execute Phase 1 (ingestion + assessment), then Phase 2 (section reviews), and finally open the summary report."
}
```

**Test Cases:**
1. ✅ "Run full review" → `["run_phase1", "run_phase2", "run_phase3"]`
2. ✅ "Re-run Phase 2 for Escalations" → `["run_phase2"]` with `section_scope=["Escalations"]`
3. ✅ "Apply changes 1, 2, 3, 4" → `["apply_changes"]` with `change_ids=["CHG-001", "CHG-002", "CHG-003", "CHG-004"]`
4. ✅ "Apply only high severity" → `["apply_changes"]` with `severity_filter="high"`
5. ✅ "Show me the doc summary" → `["open_artifact"]` with `path="/phase1/doc_summary.md"`

#### **2.3.4 State Summary & Failure Handling**

- **State Snapshot:** Planner receives `phase1_status`, `phase2_status`, `phase3_status` with values `pending|running|success|failed`, plus `last_error` text. If a phase previously failed, the planner must propose rerunning that phase before advancing.
- **Locks:** Each run stores `locked_by` with session ID. Planner refuses to execute if another session holds the lock; agent console prompts user to resume or clone the run.
- **Failure Policy:** If a plan step raises an exception (tool failure, LLM error), execution stops immediately. Backend emits `plan_step_failed` with node name, error, and suggestion. Transient LLM errors are retried once (exponential backoff) before surfacing in UI.
- **Command Safety:** `open_artifact` only accepts paths beneath the run’s VFS root (`/original`, `/phase1`, `/phase2`, `/changes`, `/versions`). Backend sanitizes inputs and rejects attempts to traverse upward.
- **Template Selection Commands:** Planner supports `use_template:<template_id>` style commands (e.g., “Use policy template”). When invoked, planner sets `state.template_meta.template_id` and confirms the change back to the user before running additional phases.

---

### **2.4 Front-End Architecture (VS Code Web)**

#### **2.4.1 VS Code Web as Front-End Shell**

The UI is implemented as a **self-hosted VS Code Web** instance:

* **Left (Explorer):** Shows all artifacts as files/folders in a virtual file system
* **Middle (Editor/Diff):** Shows selected file content, supports diff view for version comparison
* **Right (Agent Console):** Custom WebView panel with chat interface + execution trace

**Hosting Model:** VS Code Web OSS bundle is served directly from the Flask server (e.g., `/doc-review`) using the same authentication cookie. Flask proxies `/socket.io/` and `/vfs/*` so the IDE and backend share origin/credentials. Deployment doc must cover nginx reverse proxy, TLS, and cache headers.

**Why VS Code Web?**
- Users already familiar with VS Code interface
- Built-in diff viewer, syntax highlighting, JSON/Markdown rendering
- Extensible via custom file system providers and WebView panels
- Professional, production-ready UI out of the box

#### **2.4.2 Virtual File System (VFS)**

We expose the backend state as a **virtual file system** in VS Code:

**Example VFS Structure:**
```text
/original/
  document.md                    # Original uploaded document
/phase1/
  doc_summary.md                 # Executive summary (5-7 sentences)
  toc_review.json                # TOC & structure review
  template_fitness.json          # Template fitness report
  section_strategy.json          # Section strategy & next steps
/phase2/
  sections/
    Overview.md                  # Extracted section text
    Scope.md
    Governance.md
    Escalations.md
    ...
  reviews/
    Overview_review.json         # Section review with issues
    Governance_review.json
    Escalations_review.json
    ...
  summary_report.json            # Overall Phase 2 summary
/changes/
  suggested_changes.json         # All suggested changes (list)
  change_CHG-001.json            # Individual change details
  change_CHG-002.json
  ...
/versions/
  v1_original.md                 # Original document
  v2_after_changes.md            # Document after applying changes
  diff_v1_v2.md                  # Human-readable diff
```

**VFS Implementation (TypeScript):**

The VFS implements the VS Code File System API and now **supports editing**:

```typescript
class DocReviewVFS implements vscode.FileSystemProvider {
  async stat(uri: vscode.Uri): Promise<vscode.FileStat> {
    return fetchJSON(`/vfs/stat?path=${encodeURIComponent(uri.path)}`);
  }
  
  async readFile(uri: vscode.Uri): Promise<Uint8Array> {
    const resp = await fetch(`/vfs/file?path=${encodeURIComponent(uri.path)}`);
    const buf = await resp.arrayBuffer();
    return new Uint8Array(buf);
  }
  
  async writeFile(uri: vscode.Uri, content: Uint8Array, options: { create: boolean; overwrite: boolean }): Promise<void> {
    await fetch(`/vfs/file`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        path: uri.path,
        data: Buffer.from(content).toString('utf-8'),
        create: options.create,
        overwrite: options.overwrite,
      }),
      credentials: 'include',
    });
  }
  
  async readDirectory(uri: vscode.Uri): Promise<[string, vscode.FileType][]> {
    return fetchJSON(`/vfs/tree?path=${encodeURIComponent(uri.path)}`);
  }
}
```

**Backend VFS Endpoints:**

```python
@app.route('/vfs/stat', methods=['GET'])
def vfs_stat():
    path = request.args.get('path')
    # Return file metadata (size, mtime, type)
    
@app.route('/vfs/file', methods=['GET'])
def vfs_read_file():
    path = request.args.get('path')
    # Return file content from AgentState
    
@app.route('/vfs/tree', methods=['GET'])
def vfs_read_directory():
    path = request.args.get('path')
    # Return list of files/folders in directory
```

```python
@app.route('/vfs/file', methods=['PATCH'])
@login_required
def vfs_write_file():
    payload = request.get_json(force=True)
    path = sanitize_vfs_path(payload['path'])
    data = payload['data']
    write_content_to_state(run_id=current_run(), path=path, data=data)
    emit_ws({
        "type": "vfs_file_updated",
        "run_id": current_run(),
        "path": path,
        "updated_by": current_user.username,
    })
    return {"status": "ok"}
```

All write operations persist back into the AgentState (or versioned file) so edits are reflected in subsequent phases.

#### **2.4.3 WebSocket → VFS Updates**

WebSocket events (`node_completed`, `vfs_file_updated`) are used by the front-end to:

* **Refresh files** in the explorer when new artifacts are created
* **Automatically open** key artifacts (e.g., `section_strategy.md` after Phase 1)
* **Trigger diff views** (`vscode.diff` between `/versions/v1_original.md` and `/versions/v2_after_changes.md`)

**Example WebSocket Flow:**

1. User: "Run full review"
2. Backend: Executes Phase 1
3. Backend emits: `{"type": "vfs_file_updated", "path": "/phase1/doc_summary.md"}`
4. Front-end: Refreshes `/phase1` folder in explorer
5. Backend emits: `{"type": "open_artifact", "path": "/phase1/doc_summary.md"}`
6. Front-end: Opens `doc_summary.md` in editor tab

#### **2.4.4 Agent Console (Custom WebView Panel)**

The Agent Console is a custom WebView panel on the right side of VS Code:

**Features:**
- **Chat Interface:** Text input for natural language commands
- **Message History:** Shows user messages + agent responses
- **Execution Trace:** Real-time stream of node executions
  - `✅ phase1_doc_summary (842ms)`
  - `✅ phase2_section_reviews (1.2s)`
  - `⏳ apply_changes (running...)`
- **Clickable Entries:** Clicking a trace entry opens the relevant artifact

**Implementation:**
```typescript
// WebView HTML content
const agentConsoleHTML = `
  <div id="chat-history"></div>
  <div id="execution-trace"></div>
  <input id="chat-input" placeholder="Ask the agent..." />
`;

// WebSocket listener
socket.on('node_completed', (event) => {
  const trace = document.getElementById('execution-trace');
  trace.innerHTML += `<div>✅ ${event.label} (${event.duration_ms}ms)</div>`;
});
```

**Editable Workflow:** Users can modify artifacts (e.g., tweak `doc_summary.md`, adjust `suggested_changes.json`). After saving, backend emits `vfs_file_updated`, other sessions refresh automatically, and a toast confirms the update.

#### **2.4.5 Download & Export**

- **Download Latest Version:** A UI command (`Doc Review: Download Latest Version`) calls `GET /vfs/file?path=/versions/v{latest}.md` (or `.pdf` once rendering is added). Users can also right-click files in Explorer and select “Download”.
- **Version History:** Every apply operation saves a new entry (`/versions/v{n}.md`). Diff view compares `v{n-1}` vs `v{n}` by default, with an option to diff against `v1_original`.
- **Getting Started README:** `/original/README.md` explains the workflow, key commands, and sample agent prompts. It opens automatically when a new user loads the IDE for the first time.

---

## **3. Agent State Schema**

### **3.1 Core State Structure**

```python
class AgentState(TypedDict, total=False):
    # Identity & Control
    run_id: str                          # UUID for this review run
    doc_id: str                          # Stable document identifier
    control: str                         # Current phase/node label
    last_node: Optional[str]
    errors: List[str]
    
    # Document Metadata
    doc_meta: {
        "doc_title": str,
        "doc_source": Literal["upload", "sharepoint", "s3"],
        "page_count": int,
        "version": int
    }
    
    # Document Structure (from MCP tools)
    structure: {
        "raw_text": str,                 # Full markdown text
        "pages": List[dict],             # [{"page_num": 1, "text": "..."}]
        "headings": List[dict],          # [{"level": "H2", "title": "...", "page": 3}]
        "toc_detected": bool,
        "toc_entries": List[dict]        # [{"title": "...", "page_hint": 5}]
    }
    
    # Template Configuration
    template_meta: {
        "template_id": str,              # "policy_template"
        "template_text": str,            # Full template content
        "template_categories": List[str], # ["Overview", "Scope", ...]
        "max_section_words": int         # 500 (for change application safety)
    }
    
    # Phase 1 Outputs (LLM-generated)
    phase1: {
        "doc_summary": DocSummaryReport,
        "toc_review": TocReviewReport,
        "template_fitness_report": TemplateFitnessSummary,
        "section_strategy": SectionStrategyReport
    }
    
    # Phase 2 Outputs
    phase2: {
        "chunks": Dict[str, dict],       # section_title -> {text, page_range}
        "reviews": Dict[str, dict],      # section_title -> review object
        "summary_report": dict
    }
    
    # Change Management
    changes: {
        "suggested_changes": List[SuggestedChange],
        "applied_change_ids": List[str],
        "change_selection_plan": dict,
        "new_raw_text": Optional[str]
    }
    
    # User Interaction
    user_interaction: {
        "user_selected_section_strategy": bool,
        "user_change_instruction": Optional[str],
        "selected_chunking_option": Optional[str]
    }
```

### **3.2 Key Data Models**

#### **DocSummaryReport**
```python
{
    "summary": str,                      # 5-7 sentences
    "document_type": str,                # "Policy" | "Procedure" | "Playbook"
    "purpose": str,
    "audience": str,
    "themes": List[str],                 # ["Governance", "Escalations", ...]
    "confidence": "high|medium|low"
}
```

#### **TocReviewReport**
```python
{
    "toc_present": bool,
    "toc_label": Optional[str],          # "Table of Contents" or null
    "structure_score": "excellent|good|fair|poor",
    "entries": List[TocEntry],
    "observations": List[str],           # Strengths
    "gaps": List[str]                    # Issues to fix
}
```

#### **TemplateFitnessSummary**
```python
{
    "template_id": str,
    "overall_alignment": "excellent|good|fair|poor",
    "categories": List[{
        "name": str,                     # "Escalations"
        "coverage": "complete|partial|missing",
        "effort": "none|low|medium|high",
        "gaps": List[str],
        "actions": List[str]             # Operational fixes
    }],
    "narrative": str                     # ≤500 words
}
```

#### **SectionStrategyReport**
```python
{
    "verdict": "ready|needs_improvement",
    "rationale": str,
    "recommended_section_level": "h1|h2|h3|h4|h5",
    "fallback_levels": List[str],
    "estimated_sections": Optional[int],
    "next_steps": List[str]              # Actionable instructions
}
```

#### **SuggestedChange**
```python
{
    "id": str,                           # "CHG-001"
    "index": int,                        # 1..N (for user reference)
    "section_title": str,
    "page_hint": int,
    "location_instruction": str,         # "Page 5, second paragraph"
    "original_text": str,                # Exact text to replace
    "suggested_text": str,               # Replacement text
    "severity": "low|medium|high",
    "type": "grammar|clarity|structural|missing_content|terminology|tone|compliance_precision",
    "reason": str,                       # Justification
    "status": "pending|applied|ignored"
}
```

---

## **4. Phase 0: Document Ingestion**

### **4.1 MCP Tool 1: `tool_load_document`**

**Purpose:** Read uploaded file and extract raw text + pages.

**Inputs:**
```python
{
    "doc_uri": str,                      # File path or URL
    "doc_source": str                    # "upload" | "sharepoint" | "s3"
}
```

**Outputs:**
```python
{
    "raw_text": str,                     # Full markdown-converted text
    "pages": List[{"page_num": int, "text": str}],
    "doc_title": str,                    # Extracted from filename or metadata
    "page_count": int
}
```

**Test Cases:**
1. ✅ PDF upload (`Template_Version.pdf`) → 11 pages, ~4600 words
2. ✅ DOCX upload → Correct page breaks
3. ✅ Markdown upload → Pass-through without conversion
4. ✅ Unsupported format → Error with clear message
5. ✅ Missing file → 404 error

---

### **4.2 MCP Tool 2: `tool_extract_headings`**

**Purpose:** Parse document structure and extract heading hierarchy.

**Inputs:**
```python
{
    "raw_text": str,
    "pages": List[dict]
}
```

**Outputs:**
```python
{
    "headings": List[{
        "level": "H1|H2|H3|H4|H5",
        "title": str,
        "page": int,
        "char_start": int,
        "char_end": int,
        "numbering": Optional[str]       # "3.1", "3.2", etc.
    }]
}
```

**Test Cases:**
1. ✅ Multi-level hierarchy (H1→H3) → Correct nesting
2. ✅ Numbered headings ("3. Governance") → Preserve numbering
3. ✅ Flat structure (all H1) → Detect and flag
4. ✅ Mixed markdown/Word headings → Normalize to H1-H5
5. ✅ No headings → Return empty list

---

### **4.3 MCP Tool 3: `tool_detect_toc`**

**Purpose:** Identify explicit table of contents or infer from headings.

**Inputs:**
```python
{
    "raw_text": str,
    "headings": List[dict]
}
```

**Outputs:**
```python
{
    "toc_detected": bool,
    "toc_entries": List[{
        "title": str,
        "level": "H1|H2|H3",
        "page_hint": Optional[int],
        "numbering": Optional[str]
    }]
}
```

**Test Cases:**
1. ✅ Explicit TOC ("Table of Contents" heading) → `toc_detected=True`
2. ✅ No explicit TOC → Infer from H1/H2 headings
3. ✅ TOC with page numbers → Extract page hints
4. ✅ TOC without page numbers → `page_hint=null`
5. ✅ Misordered TOC → Flag in Phase 1 review

---

## **5. Phase 1: Holistic Assessment**

### **5.1 LLM Node 1: `phase1_doc_summary_llm`**

**Prompt File:** `external/doc_review/prompts/phase1_doc_summary.md`

**Purpose:** Generate executive summary proving the LLM "read" the document.

**Inputs:**
- `raw_text` (full doc if ≤10 pages, else first 5 pages)
- `page_count`, `doc_title`

**Prompt Behavior:**
- Use 5-7 punchy sentences
- Reference concrete sections (e.g., "Section 3 outlines...")
- Identify document type, purpose, audience
- Extract major themes

**Outputs:** `phase1.doc_summary` (DocSummaryReport)

**Test Cases:**

| Document | Expected `document_type` | Expected `confidence` | Expected Themes |
|----------|--------------------------|----------------------|-----------------|
| Template_Version.pdf | "Policy" | "high" | ["Governance", "Escalations", "Compliance"] |
| Middle_Version.pdf | "Policy" | "medium" | ["Governance", "Controls"] (missing Escalations) |
| Shitty_Version.pdf | "Procedure" | "low" | ["Operations"] (poor structure) |

**Business Validation:**
- ✅ Summary references specific sections by name
- ✅ Themes match template categories
- ✅ Confidence correlates with structure quality

---

### **5.2 LLM Node 2: `phase1_toc_review_llm`**

**Prompt File:** `external/doc_review/prompts/phase1_toc_review.md`

**Purpose:** Evaluate TOC quality and structural coherence.

**Inputs:**
- `headings`, `toc_detected`, `toc_entries`
- First 5 pages (for context)

**Prompt Behavior:**
- Identify TOC presence and labeling
- Assess ordering, hierarchy, flow
- Flag missing/misplaced sections
- Score structure: excellent/good/fair/poor

**Outputs:** `phase1.toc_review` (TocReviewReport)

**Test Cases:**

| Document | `toc_present` | `structure_score` | Key Gaps |
|----------|---------------|-------------------|----------|
| Template_Version.pdf | `true` | "excellent" | None |
| Middle_Version.pdf | `true` | "good" | "Escalations section buried mid-doc" |
| Shitty_Version.pdf | `false` | "poor" | "No TOC, flat H1 structure, Appendix before Escalations" |

**Business Validation:**
- ✅ Detects missing mandatory sections (Escalations, Appendix)
- ✅ Flags misordered sections (Appendix before Escalations)
- ✅ Identifies inconsistent heading levels

---

### **5.3 LLM Node 3: `phase1_template_fitness_llm`**

**Prompt File:** `external/doc_review/prompts/phase1_template_fitness.md`

**Purpose:** Compare document concepts to policy template categories.

**Inputs:**
- `raw_text` (full doc if ≤10 pages)
- `template_text` (from `data/docreview/policy_template`)
- `template_categories` (9 sections)

**Prompt Behavior:**
- **Match concepts, not headings** (e.g., "Responsibilities" → "Roles & Responsibilities")
- Group related sections into categories
- For each category:
  - `coverage`: complete/partial/missing
  - `effort`: none/low/medium/high
  - `gaps`: Specific deficiencies
  - `actions`: Operational fixes (e.g., "Add approval matrix for severity 1-3")
- Provide ≤500-word narrative on overall fit, tone gaps, priority issues

**Outputs:** `phase1.template_fitness_report` (TemplateFitnessSummary)

**Test Cases:**

| Document | `overall_alignment` | Escalations Category | Effort |
|----------|---------------------|----------------------|--------|
| Template_Version.pdf | "excellent" | `coverage: complete` | "none" |
| Middle_Version.pdf | "good" | `coverage: partial`, gaps: ["Missing approval ladder"] | "medium" |
| Shitty_Version.pdf | "poor" | `coverage: missing` | "high" |

**Business Validation:**
- ✅ Detects concept coverage even when section titles differ
- ✅ Identifies tone gaps (passive vs. prescriptive language)
- ✅ Provides actionable remediation steps
- ✅ Narrative highlights priority issues (e.g., "Escalations lack severity classification")

---

### **5.4 LLM Node 4: `phase1_section_strategy_llm`**

**Prompt File:** `external/doc_review/prompts/phase1_section_strategy.md`

**Purpose:** Synthesize Phase 1 findings and recommend next steps.

**Inputs:**
- `phase1.doc_summary`
- `phase1.toc_review`
- `phase1.template_fitness_report`

**Prompt Behavior:**
- Provide verdict: ready / needs_improvement
- Justify verdict with evidence from prior reports
- Recommend section hierarchy (H2 primary, H3 for subsections)
- Estimate section count
- List top 3-5 actionable next steps (operational wording)

**Outputs:** `phase1.section_strategy` (SectionStrategyReport)

**Test Cases:**

| Document | `verdict` | `recommended_section_level` | Key Next Steps |
|----------|-----------|----------------------------|----------------|
| Template_Version.pdf | "ready" | "h2" | ["Proceed to section extraction"] |
| Middle_Version.pdf | "needs_improvement" | "h2" | ["Rename Section 4 to 'Escalation Matrix'", "Add approval ladder with severity 1-3"] |
| Shitty_Version.pdf | "needs_improvement" | "h2" (restructure) | ["Create explicit TOC", "Split Operations into Procedures + Escalations", "Add Governance section with ownership model"] |

**Business Validation:**
- ✅ Verdict aligns with template fitness score
- ✅ Next steps are specific and actionable (not abstract)
- ✅ Section strategy is realistic (estimated count matches document size)

---

## **6. Phase 2: Section-Level Deep Review**

### **6.1 User Confirmation Step**

**UI Action:** User reviews Phase 1 reports and confirms section strategy via dropdown:
- "Use Template Strategy (H2/H3)"
- "Use Existing Structure (H2 only)"
- "Custom" (manual section selection)

**Backend:** Sets `user_interaction.user_selected_section_strategy = True` and `control = "phase2_extract_sections"`

---

### **6.2 MCP Tool 4: `tool_extract_section_by_headings`**

**Purpose:** Extract section text using heading boundaries.

**Inputs:**
```python
{
    "section_title": str,                # "Escalations"
    "headings": List[dict],
    "pages": List[dict]
}
```

**Logic:**
- Find heading by fuzzy match on title
- Extract from heading start to next heading of same/higher level
- Handle nested subsections (H3 under H2)

**Outputs:**
```python
{
    "section_text": str,
    "page_range": [int, int],
    "char_range": [int, int],
    "method": "headings"
}
```

**Test Cases:**
1. ✅ Extract "Escalations" (H2) → Includes all H3 subsections
2. ✅ Handle nested "Escalations > Severity Levels" (H3)
3. ✅ Missing section → Return empty with warning
4. ✅ Ambiguous title match → Choose closest match

---

### **6.3 MCP Tool 5: `tool_extract_section_by_toc`**

**Purpose:** Extract section text using TOC page hints.

**Inputs:**
```python
{
    "section_title": str,
    "toc_entries": List[dict],
    "pages": List[dict]
}
```

**Logic:**
- Find TOC entry by title match
- Use `page_hint` as start page
- Extract until next TOC entry's page

**Outputs:** Same as `tool_extract_section_by_headings`

**Test Cases:**
1. ✅ Extract "Escalations" using TOC page 7 → Correct boundaries
2. ✅ TOC missing page numbers → Fall back to heading method
3. ✅ TOC page hint incorrect → Validator LLM detects mismatch

---

### **6.4 LLM Node 5: `phase2_section_extraction_validator_llm`**

**Purpose:** Choose best extraction method or merge candidates.

**Inputs:**
- Candidate A (from `tool_extract_section_by_headings`)
- Candidate B (from `tool_extract_section_by_toc`)
- `section_title`, `section_strategy`

**Prompt Behavior:**
- Compare boundaries, completeness, title match
- Choose "headings" / "toc" / "merged"
- Flag issues (incomplete, wrong boundaries)

**Outputs:**
```python
{
    "is_correct": bool,
    "chosen_method": "headings|toc|merged",
    "boundary_check": "perfect|ok|incomplete",
    "issues": List[str],
    "final_section_text": str,
    "page_range": [int, int]
}
```

**Test Cases:**
1. ✅ Both candidates match → Choose "headings" (preferred)
2. ✅ Heading boundaries incomplete → Choose "toc"
3. ✅ Both incomplete → Merge and flag for review
4. ✅ Title mismatch → Retry with alternate title

---

### **6.5 LLM Node 6: `phase2_section_review_llm`**

**Purpose:** Deep template-aligned review of each section.

**Inputs:**
- `section_text` (from validated extraction)
- `template_text` (relevant category description)
- `section_strategy` (expected purpose)
- `max_section_words` (500 for safety)

**Prompt Behavior:**
- Evaluate section vs. template expectations
- Produce `fit`: none/partial/good
- Assign `severity`: low/medium/high
- Generate **issue list** with line-level suggestions:
  - `issue_id`: "GOV-001"
  - `title`: "Advisable vs must"
  - `severity`: "high"
  - `type`: "compliance_precision"
  - `location_instruction`: "Page 5, second paragraph"
  - `original_text`: "It is advisable that Market Risk review limits periodically."
  - `suggested_text`: "Market Risk must review limits at least quarterly."
  - `reason`: "Template requires mandatory 'must' language for governance obligations."

**Outputs:**
```python
{
    "section_title": str,
    "fit": "none|partial|good",
    "severity": "low|medium|high",
    "issues": List[SuggestedChange],
    "improvement_guidance": List[str]
}
```

**Test Cases:**

| Section | Document | Expected Issues |
|---------|----------|-----------------|
| Escalations | Template_Version.pdf | None (fit: "good") |
| Escalations | Middle_Version.pdf | ["Missing severity ladder", "Approval matrix incomplete"] (fit: "partial", severity: "medium") |
| Escalations | Shitty_Version.pdf | ["Section missing entirely"] (fit: "none", severity: "high") |

**Business Validation:**
- ✅ Detects passive voice ("advisable" → "must")
- ✅ Identifies missing mandatory elements (approval matrix)
- ✅ Flags unclear ownership ("team" → "Market Risk Manager")
- ✅ Suggests specific text replacements (not abstract comments)

---

### **6.6 LLM Node 7: `phase2_summary_llm`**

**Purpose:** Aggregate section reviews into overall assessment.

**Inputs:**
- `phase2.reviews` (all section reviews)
- `changes.suggested_changes` (flattened issues)

**Prompt Behavior:**
- Generate section heatmap (severity by section)
- Identify systemic gaps (repeated issues across sections)
- Provide overall posture: ready / needs_work / needs_overhaul
- Short narrative for UI (≤300 words)

**Outputs:**
```python
{
    "overall_posture": "ready|needs_work|needs_overhaul",
    "section_heatmap": Dict[str, "low|medium|high"],
    "systemic_gaps": List[str],           # ["Passive voice throughout", "Missing ownership"]
    "narrative": str,
    "total_issues": int,
    "high_severity_count": int
}
```

**Test Cases:**
1. ✅ Template_Version.pdf → `overall_posture: "ready"`, 0 high-severity issues
2. ✅ Middle_Version.pdf → `overall_posture: "needs_work"`, 3 high-severity issues (Escalations, Governance)
3. ✅ Shitty_Version.pdf → `overall_posture: "needs_overhaul"`, 12 high-severity issues

---

## **7. Phase 3: Change Selection & Application**

### **7.1 User Instruction Step**

**UI Action:** User types free-text instruction:
- "Apply all"
- "Apply only high severity"
- "Apply 1, 2, 3, 4"
- "Apply all changes in Governance and Escalations"

**Backend:** Sets `user_interaction.user_change_instruction` and `control = "change_selection_intent"`

---

### **7.2 LLM Node 8: `change_selection_intent_llm`**

**Purpose:** Interpret user instruction and select change IDs.

**Inputs:**
- `user_change_instruction` (free text)
- `changes.suggested_changes` (with IDs, severity, index)

**Prompt Behavior:**
- Parse instruction intent
- Select matching change IDs
- **Never edit text** (only select IDs)

**Outputs:**
```python
{
    "apply_mode": "all|by_ids|by_severity|by_section",
    "change_ids_to_apply": List[str],    # ["CHG-001", "CHG-002", ...]
    "rationale": str                     # "User said '1,2,3,4 only'"
}
```

**Test Cases:**
1. ✅ "Apply all" → Select all pending changes
2. ✅ "Apply 1, 2, 3, 4" → Select CHG-001 through CHG-004
3. ✅ "Apply only high severity" → Select changes where `severity="high"`
4. ✅ "Apply Governance changes" → Select changes where `section_title="Governance"`
5. ✅ Ambiguous instruction → Ask for clarification

---

### **7.3 MCP Tool 6: `tool_apply_changes_deterministic`**

**Purpose:** Apply selected changes using exact text replacement.

**Inputs:**
```python
{
    "raw_text": str,
    "suggested_changes": List[SuggestedChange],
    "change_ids_to_apply": List[str]
}
```

**Logic:**
1. Group changes by section (for safety)
2. For each section (≤500 words):
   - Find `original_text` in section
   - Replace with `suggested_text`
   - Verify replacement succeeded
3. Concatenate updated sections

**Outputs:**
```python
{
    "new_raw_text": str,
    "updated_sections": List[str],
    "applied_change_ids": List[str],
    "failed_changes": List[{"id": str, "reason": str}]
}
```

**Test Cases:**
1. ✅ Apply single change → Correct replacement
2. ✅ Apply multiple changes in same section → All replaced
3. ✅ Original text not found → Add to `failed_changes`
4. ✅ Section >500 words → Split and apply safely
5. ✅ Overlapping changes → Apply in order, detect conflicts

**Business Validation:**
- ✅ Only targeted text is changed (no LLM hallucinations)
- ✅ Surrounding text preserved exactly
- ✅ Failed changes reported with reason

---

### **7.4 LLM Node 9: `apply_changes_verifier_llm` (Optional)**

**Purpose:** Verify no unintended edits were made.

**Inputs:**
- `original_section`
- `updated_section`
- `applied_changes` (for this section)

**Prompt Behavior:**
- Compare original vs. updated
- Verify only described replacements occurred
- Flag any extra edits

**Outputs:**
```python
{
    "verification_status": "pass|fail",
    "issues": List[str]
}
```

---

## **8. Orchestrator & Control Flow**

### **8.1 Node Registry**

```python
NODE_REGISTRY = {
    # Phase 0 – Ingestion
    "load_document": tool_load_document,
    "extract_headings": tool_extract_headings,
    "detect_toc": tool_detect_toc,
    
    # Phase 1 – Holistic Assessment
    "phase1_doc_summary": node_phase1_doc_summary_llm,
    "phase1_toc_review": node_phase1_toc_review_llm,
    "phase1_template_fitness": node_phase1_template_fitness_llm,
    "phase1_section_strategy": node_phase1_section_strategy_llm,
    
    # Phase 2 – Section Extraction & Review
    "phase2_extract_sections": orchestrator_phase2_extract_sections,  # Loop wrapper
    "phase2_section_reviews": orchestrator_phase2_section_reviews,    # Loop wrapper
    "phase2_summary": node_phase2_summary_llm,
    
    # Phase 3 – Change Selection & Application
    "change_selection_intent": node_change_selection_intent_llm,
    "apply_changes": tool_apply_changes_deterministic,
    "verify_changes": node_apply_changes_verifier_llm,
}
```

### **8.2 Control Flow Sequence**

```
1. load_document → extract_headings
2. extract_headings → detect_toc
3. detect_toc → phase1_doc_summary
4. phase1_doc_summary → phase1_toc_review
5. phase1_toc_review → phase1_template_fitness
6. phase1_template_fitness → phase1_section_strategy
7. phase1_section_strategy → await_section_strategy_confirmation (USER INPUT)
8. [User confirms] → phase2_extract_sections
9. phase2_extract_sections → phase2_section_reviews
10. phase2_section_reviews → phase2_summary
11. phase2_summary → await_change_instruction (USER INPUT)
12. [User provides instruction] → change_selection_intent
13. change_selection_intent → apply_changes
14. apply_changes → verify_changes (optional)
15. verify_changes → completed
```

### **8.3 Orchestrator Main Loop**

```python
def orchestrate(state: AgentState) -> AgentState:
    while True:
        control = state["control"]
        
        # Pause for user input
        if control in ["await_section_strategy_confirmation",
                       "await_change_instruction"]:
            break
        
        # Terminal states
        if control in ["completed", "failed"]:
            break
        
        # Execute node
        node_fn = NODE_REGISTRY.get(control)
        if not node_fn:
            state["errors"].append(f"Unknown control: {control}")
            state["control"] = "failed"
            break
        
        state = run_node(control, state)  # Wraps node call with WS events
    
    return state
```

---

### **8.4 Agent-Orchestrator Integration**

The **DocReviewAgent** sits on top of `orchestrate()` and provides the natural language interface:

```python
def handle_user_message(run_id: str, user_message: str) -> AgentState:
    """
    Main entrypoint for autonomous agent behavior.
    Accepts natural language command, plans actions, executes them.
    """
    state = load_state(run_id)
    
    # 1. Acquire lock & call planner LLM to interpret user intent
    acquire_run_lock(run_id, current_session_id())
    state_summary = summarize_state_for_planner(state)
    plan = agent_planner_llm(user_message, state_summary)
    
    # Emit plan to UI
    emit_ws({
        "type": "agent_plan_generated",
        "run_id": run_id,
        "plan": plan["plan_steps"],
        "explanation": plan["explanation"]
    })
    
    # 2. Execute plan steps sequentially
    for step in plan["plan_steps"]:
        action = step["action"]
        args = step.get("args", {})
        
        try:
            if action == "run_phase1":
                state = run_phase1(state)
            elif action == "run_phase2":
                section_scope = args.get("section_scope", None)
                state = run_phase2(state, section_scope=section_scope)
            elif action == "run_phase3":
                state = run_phase3(state)
            elif action == "rerun_phase1":
                state = rerun_phase1(state)
            elif action == "rerun_phase2":
                section_scope = args.get("section_scope", None)
                state = rerun_phase2(state, section_scope=section_scope)
            elif action == "rerun_section":
                section_title = args.get("section_title")
                state = rerun_section(state, section_title)
            elif action == "apply_changes":
                change_ids = args.get("change_ids", None)
                severity_filter = args.get("severity_filter", None)
                sections = args.get("sections", None)
                state = run_apply_changes(state, change_ids, severity_filter, sections)
            elif action == "open_artifact":
                path = args.get("path")
                emit_ws({
                    "type": "open_artifact",
                    "run_id": run_id,
                    "path": path
                })
            elif action == "use_template":
                template_id = args.get("template_id", "policy_template")
                state["template_meta"]["template_id"] = template_id
                emit_ws({
                    "type": "template_changed",
                    "run_id": run_id,
                    "template_id": template_id
                })
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as exc:
            emit_ws({
                "type": "plan_step_failed",
                "run_id": run_id,
                "action": action,
                "error": str(exc)
            })
            release_run_lock(run_id, current_session_id())
            raise
    
    # 3. Save and return
    save_state(run_id, state)
    update_vfs(run_id, state)  # Sync VFS with new artifacts
    release_run_lock(run_id, current_session_id())
    
    return state


def run_phase1(state: AgentState) -> AgentState:
    """Execute Phase 0 + Phase 1 using orchestrator."""
    state["control"] = "load_document"
    state = orchestrate(state)  # Runs until phase1_section_strategy completes
    
    # Emit VFS updates
    emit_vfs_updates(state, [
        "/phase1/doc_summary.md",
        "/phase1/toc_review.json",
        "/phase1/template_fitness.json",
        "/phase1/section_strategy.json"
    ])
    
    return state


def run_phase2(state: AgentState, section_scope=None) -> AgentState:
    """Execute Phase 2 using orchestrator."""
    if section_scope:
        state["user_interaction"]["selected_section_scope"] = section_scope
    
    state["control"] = "phase2_extract_sections"
    state = orchestrate(state)  # Runs until phase2_summary completes
    
    # Emit VFS updates
    sections = state["phase2"]["chunks"].keys()
    vfs_paths = [f"/phase2/sections/{s}.md" for s in sections]
    vfs_paths += [f"/phase2/reviews/{s}_review.json" for s in sections]
    vfs_paths.append("/phase2/summary_report.json")
    emit_vfs_updates(state, vfs_paths)
    
    return state


def run_apply_changes(state: AgentState, change_ids=None, severity_filter=None, sections=None) -> AgentState:
    """Apply selected changes using orchestrator."""
    # Filter changes based on criteria
    all_changes = state["changes"]["suggested_changes"]
    
    if change_ids:
        selected_changes = [c for c in all_changes if c["id"] in change_ids]
    elif severity_filter:
        selected_changes = [c for c in all_changes if c["severity"] == severity_filter]
    elif sections:
        selected_changes = [c for c in all_changes if c["section_title"] in sections]
    else:
        selected_changes = all_changes  # Apply all
    
    state["user_interaction"]["user_change_instruction"] = f"Apply {len(selected_changes)} changes"
    state["changes"]["selected_change_ids"] = [c["id"] for c in selected_changes]
    
    state["control"] = "apply_changes"
    state = orchestrate(state)  # Runs apply_changes + verify_changes
    
    # Emit VFS updates
    emit_vfs_updates(state, [
        "/versions/v2_after_changes.md",
        "/versions/diff_v1_v2.md"
    ])
    
    # Auto-open diff view
    emit_ws({
        "type": "open_diff",
        "run_id": state["run_id"],
        "left": "/versions/v1_original.md",
        "right": "/versions/v2_after_changes.md"
    })
    
    return state


def summarize_state_for_planner(state: AgentState) -> dict:
    """Create high-level summary for planner LLM."""
    return {
        "doc_id": state.get("doc_id"),
        "phase1_done": state.get("phase1") is not None,
        "phase2_done": state.get("phase2") is not None,
        "phase3_done": state.get("changes", {}).get("new_raw_text") is not None,
        "total_changes": len(state.get("changes", {}).get("suggested_changes", [])),
        "high_severity_count": len([c for c in state.get("changes", {}).get("suggested_changes", []) if c["severity"] == "high"])
    }


def emit_vfs_updates(state: AgentState, paths: List[str]):
    """Emit VFS file update events for front-end."""
    for path in paths:
        emit_ws({
            "type": "vfs_file_updated",
            "run_id": state["run_id"],
            "path": path
        })
```

**Key Design Points:**

1. **Thin Agent Layer:** The agent is just a wrapper around existing `orchestrate()` calls
2. **No Hidden Logic:** Every agent action maps to a known orchestrator path
3. **VFS Sync:** After each phase, VFS is updated and front-end notified
4. **Modular:** Each `run_phaseX` function is independently testable

---

## **9. WebSocket Event Streaming**

### **9.1 Event Schema**

**Node Started:**
```json
{
  "type": "node_started",
  "run_id": "RUN-123",
  "node_id": "phase1_doc_summary",
  "node_kind": "llm",
  "label": "Phase 1 – Document Summary",
  "timestamp": "2025-11-15T12:00:00Z"
}
```

**Node Completed:**
```json
{
  "type": "node_completed",
  "run_id": "RUN-123",
  "node_id": "phase1_doc_summary",
  "node_kind": "llm",
  "label": "Phase 1 – Document Summary",
  "status": "success",
  "duration_ms": 842,
  "short_result_summary": "5-7 sentence summary generated.",
  "payload_ref": "state.phase1.doc_summary"
}
```

### **9.2 Backend Wrapper**

```python
def run_node(node_id: str, state: AgentState) -> AgentState:
    emit_ws({
        "type": "node_started",
        "run_id": state["run_id"],
        "node_id": node_id,
        "node_kind": NODE_KIND[node_id],
        "label": NODE_LABEL[node_id],
        "timestamp": now_iso()
    })
    
    start_time = time.time()
    try:
        new_state = NODE_REGISTRY[node_id](state)
        status = "success"
        error_msg = None
    except Exception as e:
        new_state = state
        status = "failed"
        error_msg = str(e)
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    emit_ws({
        "type": "node_completed",
        "run_id": state["run_id"],
        "node_id": node_id,
        "status": status,
        "duration_ms": duration_ms,
        "short_result_summary": summarize_output(node_id, new_state),
        "payload_ref": f"state.{get_output_path(node_id)}",
        "error": error_msg
    })
    
    return new_state
```

---

### **9.3 IDE-Specific Event Types**

Additional events for the VS Code front-end:

**Agent Plan Generated:**
```json
{
  "type": "agent_plan_generated",
  "run_id": "RUN-123",
  "plan": [
    {"action": "run_phase1", "args": {"doc_id": "DOC-123"}},
    {"action": "run_phase2", "args": {"doc_id": "DOC-123"}}
  ],
  "explanation": "User asked to run a full review. Will execute Phase 1 (ingestion + assessment), then Phase 2 (section reviews)."
}
```

**VFS File Updated:**
```json
{
  "type": "vfs_file_updated",
  "run_id": "RUN-123",
  "path": "/phase1/doc_summary.md"
}
```

**Open Artifact in Editor:**
```json
{
  "type": "open_artifact",
  "run_id": "RUN-123",
  "path": "/phase1/section_strategy.json"
}
```

**Open Diff View:**
```json
{
  "type": "open_diff",
  "run_id": "RUN-123",
  "left": "/versions/v1_original.md",
  "right": "/versions/v2_after_changes.md"
}
```

**Front-End Handler (TypeScript):**

```typescript
socket.on('vfs_file_updated', (event) => {
  // Refresh file content in VFS
  vfs.refresh(event.path);
  
  // Update explorer tree view
  vscode.commands.executeCommand('workbench.files.action.refreshFilesExplorer');
});

socket.on('open_artifact', (event) => {
  // Open file in editor
  const uri = vscode.Uri.parse(`docreview:${event.path}`);
  vscode.window.showTextDocument(uri);
});

socket.on('open_diff', (event) => {
  // Open diff view
  const leftUri = vscode.Uri.parse(`docreview:${event.left}`);
  const rightUri = vscode.Uri.parse(`docreview:${event.right}`);
  vscode.commands.executeCommand('vscode.diff', leftUri, rightUri, 'Original ↔ After Changes');
});

socket.on('agent_plan_generated', (event) => {
  // Show plan in Agent Console
  const console = getAgentConsole();
  console.appendMessage(`🤖 Plan: ${event.explanation}`);
  event.plan.forEach((step, i) => {
    console.appendMessage(`  ${i+1}. ${step.action}(${JSON.stringify(step.args)})`);
  });
});
```

### **9.4 Observability, Logging & Metrics**

- **Structured Logs:** Every agent command is logged as JSON (`run_id`, `user`, `message`, `plan_steps`, `status`, `duration_ms`). Logs are stored under `logs/RUN_ID/agent_transcript.jsonl`.
- **Transcript Retention:** Chat history + plan outputs are persisted so auditors can replay decisions. Sensitive content is masked per compliance guidelines.
- **Metrics:** Prometheus counters/gauges track phase durations (`phase1_duration_ms`), planner accuracy (`planner_success_total` vs `planner_failure_total`), WebSocket latency, and VFS write latency.
- **Alerts:** If `plan_step_failed` fires more than 3 times in 10 minutes for the same run, emit an alert event so the UI can suggest contacting support.

---

## **10. UI/UX Design**

### **10.1 VS Code Web Layout**

```
┌──────────────────────────────────────────────────────────────────────┐
│  VS Code Web - Document Review Agent                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┬────────────────────────────┬──────────────────┐    │
│  │  Explorer   │      Editor / Diff         │  Agent Console   │    │
│  │  (Left)     │      (Middle)              │  (Right)         │    │
│  ├─────────────┼────────────────────────────┼──────────────────┤    │
│  │ 📁 original │  [doc_summary.md]          │ 💬 Chat          │    │
│  │   document.md                            │ ┌──────────────┐ │    │
│  │             │  This document is a        │ │ User: Run    │ │    │
│  │ 📁 phase1   │  comprehensive policy...   │ │ full review  │ │    │
│  │   doc_summary.md (open)                  │ └──────────────┘ │    │
│  │   toc_review.json                        │ ┌──────────────┐ │    │
│  │   template_fitness.json                  │ │ Agent: I'll  │ │    │
│  │   section_strategy.json                  │ │ run Phase 1, │ │    │
│  │             │                            │ │ then Phase 2 │ │    │
│  │ 📁 phase2   │                            │ └──────────────┘ │    │
│  │   📁 sections                            │                  │    │
│  │   📁 reviews│                            │ 📊 Trace         │    │
│  │   summary_report.json                    │ ✅ load_doc     │    │
│  │             │                            │    (842ms)       │    │
│  │ 📁 changes  │                            │ ✅ extract_head │    │
│  │   suggested_changes.json                 │    (123ms)       │    │
│  │   change_CHG-001.json                    │ ⏳ phase1_sum   │    │
│  │             │                            │    (running...)  │    │
│  │ 📁 versions │                            │                  │    │
│  │   v1_original.md                         │                  │    │
│  │   v2_after_changes.md                    │                  │    │
│  │   diff_v1_v2.md                          │                  │    │
│  └─────────────┴────────────────────────────┴──────────────────┘    │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### **10.2 Explorer (Left Panel)**

**VFS Tree Structure:**
- **`/original`** – Original uploaded document
- **`/phase1`** – Phase 1 outputs (summaries, reviews, strategy)
- **`/phase2`** – Phase 2 outputs (sections, reviews, summary)
  - `/phase2/sections` – Extracted section texts
  - `/phase2/reviews` – Per-section review reports
- **`/changes`** – Suggested changes (individual + aggregated)
- **`/versions`** – Document versions (original, after changes, diff)

**Behavior:**
- Clicking a file opens it in the editor
- Files update automatically via `vfs_file_updated` events
- JSON files render with syntax highlighting
- Markdown files render with preview

### **10.3 Editor (Middle Panel)**

**Capabilities:**
- **Text Editing:** View/edit Markdown, JSON, text files
- **Syntax Highlighting:** Automatic for JSON, Markdown
- **Diff View:** Side-by-side comparison of versions
  - Triggered via `open_diff` WebSocket event
  - Example: `/versions/v1_original.md` ↔ `/versions/v2_after_changes.md`
- **Search:** Built-in VS Code search across all artifacts

**Auto-Open Behavior:**
- After Phase 1: Opens `/phase1/section_strategy.json`
- After Phase 2: Opens `/phase2/summary_report.json`
- After apply changes: Opens diff view

### **10.4 Agent Console (Right Panel – Custom WebView)**

**Chat Interface:**
- Text input at bottom: "Ask the agent..."
- Message history shows user commands + agent responses
- Agent responses include plan explanation

**Execution Trace:**
- Real-time stream of node executions
- Format: `✅ phase1_doc_summary (842ms)`
- Clickable entries open relevant artifacts
- Color-coded:
  - ✅ Green: Success
  - ⏳ Yellow: Running
  - ❌ Red: Failed

**Example Interaction:**
```
User: Run full review on this document

Agent: I'll execute a full review workflow:
  1. run_phase1(doc_id=DOC-123)
  2. run_phase2(doc_id=DOC-123)
  3. open_artifact(path=/phase2/summary_report.json)

Execution Trace:
✅ load_document (842ms)
✅ extract_headings (123ms)
✅ detect_toc (89ms)
✅ phase1_doc_summary (1.2s)
✅ phase1_toc_review (980ms)
✅ phase1_template_fitness (1.5s)
✅ phase1_section_strategy (750ms)
⏳ phase2_extract_sections (running...)
```

### **10.5 User Guidance & Export Workflow**

- **Auto-Onboarding:** `/original/README.md` opens automatically on first visit with a 3-step quick start guide and example commands (“Run full review”, “Apply high severity changes”, “Show template fitness”).
- **Command Suggestions:** When the chat input is empty, placeholder text rotates through suggested commands. After each successful plan, the agent console proposes logical next steps (e.g., “Need to re-run Phase 2 for specific sections?”).
- **Download Button:** A toolbar action (`Download latest version`) fetches `/versions/v{latest}.md`. After download, the agent console confirms the filename and timestamp.
- **Manual Editing Guardrails:** When users edit files, the status bar shows “Editing /phase1/doc_summary.md • Auto-saved”. Attempting to edit system folders (outside the allowed whitelist) triggers an inline warning.

---

## **11. Test Strategy & Coverage**

### **11.1 Unit Tests (Per Tool/Node)**

**File:** `test/test_doc_review_tools.py`

```python
def test_tool_load_document_pdf():
    result = tool_load_document({"doc_uri": "data/docreview/Template_Version.pdf"})
    assert result["page_count"] == 11
    assert result["doc_title"] == "Template_Version"
    assert len(result["raw_text"]) > 4000

def test_tool_extract_headings_hierarchy():
    result = tool_extract_headings({"raw_text": "# H1\n## H2\n### H3"})
    assert len(result["headings"]) == 3
    assert result["headings"][0]["level"] == "H1"
    assert result["headings"][2]["level"] == "H3"

def test_tool_detect_toc_explicit():
    result = tool_detect_toc({
        "raw_text": "# Table of Contents\n1. Overview\n2. Scope",
        "headings": [...]
    })
    assert result["toc_detected"] == True
    assert result["toc_entries"][0]["title"] == "Overview"
```

**File:** `test/test_doc_review_llm_nodes.py`

```python
def test_phase1_doc_summary_template_version():
    state = load_state_after("detect_toc", doc="Template_Version.pdf")
    result = node_phase1_doc_summary_llm(state)
    
    assert result["phase1"]["doc_summary"]["document_type"] == "Policy"
    assert "Escalations" in result["phase1"]["doc_summary"]["themes"]
    assert result["phase1"]["doc_summary"]["confidence"] == "high"

def test_phase1_toc_review_shitty_version():
    state = load_state_after("detect_toc", doc="Shitty_Version.pdf")
    result = node_phase1_toc_review_llm(state)
    
    assert result["phase1"]["toc_review"]["toc_present"] == False
    assert result["phase1"]["toc_review"]["structure_score"] == "poor"
    assert "Escalations" in str(result["phase1"]["toc_review"]["gaps"])

def test_phase1_template_fitness_concept_matching():
    state = load_state_after("detect_toc", doc="Middle_Version.pdf")
    result = node_phase1_template_fitness_llm(state)
    
    # Should match "Responsibilities" to "Roles & Responsibilities"
    roles_cat = next(c for c in result["phase1"]["template_fitness_report"]["categories"] 
                     if "Roles" in c["name"])
    assert roles_cat["coverage"] in ["complete", "partial"]
```

---

### **11.2 Integration Tests (Phase-Level)**

**File:** `test/test_doc_review_phase1_e2e.py`

```python
@pytest.mark.parametrize("doc_name,expected", [
    ("Template_Version.pdf", {
        "doc_summary.confidence": "high",
        "toc_review.structure_score": "excellent",
        "template_fitness.overall_alignment": "excellent",
        "section_strategy.verdict": "ready"
    }),
    ("Middle_Version.pdf", {
        "doc_summary.confidence": "medium",
        "toc_review.structure_score": "good",
        "template_fitness.overall_alignment": "good",
        "section_strategy.verdict": "needs_improvement"
    }),
    ("Shitty_Version.pdf", {
        "doc_summary.confidence": "low",
        "toc_review.structure_score": "poor",
        "template_fitness.overall_alignment": "poor",
        "section_strategy.verdict": "needs_improvement"
    })
])
def test_phase1_full_pipeline(doc_name, expected):
    """Run full Phase 1 for each sample document."""
    agent = DocReviewAgent()
    state = agent.run_phase1(doc_path=f"data/docreview/{doc_name}")
    
    assert state["phase1"]["doc_summary"]["confidence"] == expected["doc_summary.confidence"]
    assert state["phase1"]["toc_review"]["structure_score"] == expected["toc_review.structure_score"]
    assert state["phase1"]["template_fitness"]["overall_alignment"] == expected["template_fitness.overall_alignment"]
    assert state["phase1"]["section_strategy"]["verdict"] == expected["section_strategy.verdict"]
```

---

### **11.3 End-to-End Tests**

**File:** `test/test_doc_review_e2e.py`

```python
def test_full_workflow_template_version():
    """Complete workflow: upload → Phase 1 → Phase 2 → apply changes."""
    agent = DocReviewAgent()
    
    # Phase 0 + Phase 1
    state = agent.run_phase1("data/docreview/Template_Version.pdf")
    assert state["phase1"]["section_strategy"]["verdict"] == "ready"
    
    # User confirms strategy
    state["user_interaction"]["user_selected_section_strategy"] = True
    state["control"] = "phase2_extract_sections"
    
    # Phase 2
    state = agent.run_phase2(state)
    assert len(state["phase2"]["chunks"]) == 9  # 9 template sections
    assert state["phase2"]["summary_report"]["overall_posture"] == "ready"
    assert len(state["changes"]["suggested_changes"]) == 0  # No issues
    
def test_full_workflow_middle_version():
    """Workflow with changes: upload → Phase 1 → Phase 2 → apply changes."""
    agent = DocReviewAgent()
    
    # Phase 0 + Phase 1
    state = agent.run_phase1("data/docreview/Middle_Version.pdf")
    assert state["phase1"]["section_strategy"]["verdict"] == "needs_improvement"
    
    # User confirms strategy
    state["user_interaction"]["user_selected_section_strategy"] = True
    state["control"] = "phase2_extract_sections"
    
    # Phase 2
    state = agent.run_phase2(state)
    assert len(state["changes"]["suggested_changes"]) > 0  # Has issues
    
    # User applies high-severity changes
    state["user_interaction"]["user_change_instruction"] = "Apply only high severity"
    state["control"] = "change_selection_intent"
    
    # Phase 3
    state = agent.run_phase3(state)
    high_severity_changes = [c for c in state["changes"]["suggested_changes"] if c["severity"] == "high"]
    assert len(state["changes"]["applied_change_ids"]) == len(high_severity_changes)
    assert state["changes"]["new_raw_text"] is not None
```

---

### **11.4 Agent Planner Tests (Autonomy)**

**File:** `test/test_doc_review_agent_planner.py`

```python
def test_agent_plan_full_workflow():
    """Test planner interprets 'run full review' correctly."""
    planner_input = {
        "message": "Run a full review on this document.",
        "state_summary": {"phase1_done": False, "phase2_done": False}
    }
    plan = agent_planner_llm(planner_input)
    
    actions = [step["action"] for step in plan["plan_steps"]]
    assert actions == ["run_phase1", "run_phase2", "run_phase3"]
    assert "full review" in plan["explanation"].lower()

def test_agent_plan_rerun_phase2_for_section():
    """Test planner interprets 're-run Phase 2 for Escalations' correctly."""
    planner_input = {
        "message": "Re-run Phase 2 only for the Escalations section.",
        "state_summary": {"phase1_done": True, "phase2_done": True}
    }
    plan = agent_planner_llm(planner_input)
    
    assert len(plan["plan_steps"]) == 1
    assert plan["plan_steps"][0]["action"] == "run_phase2"
    assert plan["plan_steps"][0]["args"]["section_scope"] == ["Escalations"]

def test_agent_plan_apply_changes_by_index():
    """Test planner interprets 'apply changes 1, 2, 3, 4' correctly."""
    planner_input = {
        "message": "Apply changes 1, 2, 3 and 4 only.",
        "state_summary": {"phase2_done": True, "total_changes": 10}
    }
    plan = agent_planner_llm(planner_input)
    
    assert plan["plan_steps"][0]["action"] == "apply_changes"
    assert plan["plan_steps"][0]["args"]["change_ids"] == ["CHG-001", "CHG-002", "CHG-003", "CHG-004"]

def test_agent_plan_apply_high_severity():
    """Test planner interprets 'apply only high severity' correctly."""
    planner_input = {
        "message": "Apply all high severity changes.",
        "state_summary": {"phase2_done": True, "high_severity_count": 5}
    }
    plan = agent_planner_llm(planner_input)
    
    assert plan["plan_steps"][0]["action"] == "apply_changes"
    assert plan["plan_steps"][0]["args"]["severity_filter"] == "high"

def test_agent_plan_open_artifact():
    """Test planner interprets 'show me X' correctly."""
    planner_input = {
        "message": "Show me the template fitness report.",
        "state_summary": {"phase1_done": True}
    }
    plan = agent_planner_llm(planner_input)
    
    assert plan["plan_steps"][0]["action"] == "open_artifact"
    assert "/phase1/template_fitness" in plan["plan_steps"][0]["args"]["path"]

def test_agent_plan_invalid_command():
    """Test planner handles ambiguous/invalid commands."""
    planner_input = {
        "message": "Do something with the document.",
        "state_summary": {"phase1_done": False}
    }
    plan = agent_planner_llm(planner_input)
    
    # Should ask for clarification or default to safe action
    assert "clarification" in plan["explanation"].lower() or plan["plan_steps"][0]["action"] == "run_phase1"
```

---

### **11.5 Smoke Scripts (CLI)**

- `scripts/phase1_backend_smoke.py --doc <path>` – existing script; ensure it loads env vars and prints Phase 1 outputs.
- `scripts/phase2_backend_smoke.py --doc <path> [--sections Escalations,Governance]` – runs Phase 2 end-to-end without UI.
- `scripts/phase3_backend_smoke.py --doc <path> --apply-mode high` – filters change set and applies deterministically.
- `scripts/agent_plan_smoke.py --doc <path> --message "Run full review"` – invokes `agent_planner_llm`, prints plan steps, and (optionally) executes them for headless testing.

Smoke scripts must log structured JSON so CI can parse results. They serve as the first line of defense before UI-based QA.

---

## **12. Business Validation Criteria**

### **12.1 Phase 1 Acceptance Criteria**

| Criterion | Template_Version.pdf | Middle_Version.pdf | Shitty_Version.pdf |
|-----------|----------------------|--------------------|--------------------|
| **Doc Summary** | ✅ Confidence: high, references 3+ sections | ✅ Confidence: medium, identifies partial structure | ✅ Confidence: low, flags poor organization |
| **TOC Review** | ✅ Score: excellent, no gaps | ✅ Score: good, flags "Escalations buried" | ✅ Score: poor, "No TOC, flat structure" |
| **Template Fitness** | ✅ Alignment: excellent, effort: none | ✅ Alignment: good, effort: medium, gaps in Escalations | ✅ Alignment: poor, effort: high, missing Governance |
| **Section Strategy** | ✅ Verdict: ready, recommend H2 | ✅ Verdict: needs_improvement, 3 actionable steps | ✅ Verdict: needs_improvement, 5 actionable steps |

### **12.2 Phase 2 Acceptance Criteria**

| Criterion | Template_Version.pdf | Middle_Version.pdf | Shitty_Version.pdf |
|-----------|----------------------|--------------------|--------------------|
| **Section Extraction** | ✅ 9/9 sections extracted cleanly | ✅ 8/9 sections (Escalations partial) | ✅ 3/9 sections (major restructure needed) |
| **Section Reviews** | ✅ All sections fit: "good", severity: low | ✅ Escalations fit: "partial", severity: medium | ✅ Most sections fit: "none", severity: high |
| **Suggested Changes** | ✅ 0 high-severity issues | ✅ 3 high-severity issues (Escalations, Governance) | ✅ 12 high-severity issues |
| **Overall Posture** | ✅ "ready" | ✅ "needs_work" | ✅ "needs_overhaul" |

### **12.3 Phase 3 Acceptance Criteria**

| Criterion | Success Criteria |
|-----------|------------------|
| **Change Selection** | ✅ Correctly interprets "apply all", "1,2,3,4", "high severity only" |
| **Text Replacement** | ✅ Only targeted text changed, surrounding text preserved |
| **Failed Changes** | ✅ Reports changes that couldn't be applied with reason |
| **Verification** | ✅ No unintended edits detected |

---

## **13. Sample Documents & Expected Outcomes**

### **13.1 Template_Version.pdf (Baseline)**

**Characteristics:**
- 11 pages, ~4600 words
- Explicit TOC on page 1
- 9 well-structured H2 sections matching template
- Prescriptive language ("must", "shall")
- Clear ownership and approval flows

**Expected Phase 1 Outputs:**
```json
{
  "doc_summary": {
    "confidence": "high",
    "document_type": "Policy",
    "themes": ["Overview", "Scope", "Governance", "Escalations", "Compliance"]
  },
  "toc_review": {
    "structure_score": "excellent",
    "gaps": []
  },
  "template_fitness": {
    "overall_alignment": "excellent",
    "categories": [
      {"name": "Escalations", "coverage": "complete", "effort": "none"}
    ]
  },
  "section_strategy": {
    "verdict": "ready",
    "next_steps": ["Proceed to section extraction"]
  }
}
```

**Expected Phase 2 Outputs:**
- 9/9 sections extracted
- 0 high-severity issues
- Overall posture: "ready"

---

### **13.2 Middle_Version.pdf (Partial Compliance)**

**Characteristics:**
- 8 pages, ~3400 words
- TOC present but incomplete
- 7/9 template sections present
- Missing: Escalations approval matrix, Governance ownership model
- Mixed language (some passive voice)

**Expected Phase 1 Outputs:**
```json
{
  "doc_summary": {
    "confidence": "medium",
    "themes": ["Governance", "Controls", "Compliance"]
  },
  "toc_review": {
    "structure_score": "good",
    "gaps": ["Escalations section buried mid-document"]
  },
  "template_fitness": {
    "overall_alignment": "good",
    "categories": [
      {
        "name": "Escalations",
        "coverage": "partial",
        "effort": "medium",
        "gaps": ["Missing approval ladder", "No severity classification"],
        "actions": ["Add approval matrix for severity 1-3", "Define escalation timelines"]
      }
    ]
  },
  "section_strategy": {
    "verdict": "needs_improvement",
    "next_steps": [
      "Rename Section 4 to 'Escalation Matrix'",
      "Add approval ladder with severity 1-3",
      "Define ownership in Governance section"
    ]
  }
}
```

**Expected Phase 2 Outputs:**
- 8/9 sections extracted (Escalations partial)
- 3 high-severity issues:
  1. "Advisable" → "must" (Governance)
  2. Missing approval matrix (Escalations)
  3. Unclear ownership (Governance)
- Overall posture: "needs_work"

---

### **13.3 Shitty_Version.pdf (Poor Structure)**

**Characteristics:**
- 1 page, ~300 words
- No TOC
- Flat H1 structure (no hierarchy)
- Missing: Governance, Escalations, Compliance sections
- Passive voice throughout
- No ownership or approval flows

**Expected Phase 1 Outputs:**
```json
{
  "doc_summary": {
    "confidence": "low",
    "document_type": "Procedure",
    "themes": ["Operations"]
  },
  "toc_review": {
    "structure_score": "poor",
    "gaps": ["No TOC", "Flat H1 structure", "Appendix before Escalations"]
  },
  "template_fitness": {
    "overall_alignment": "poor",
    "categories": [
      {
        "name": "Governance",
        "coverage": "missing",
        "effort": "high",
        "gaps": ["No ownership model", "No approval flows"],
        "actions": ["Create Governance section with ownership matrix"]
      },
      {
        "name": "Escalations",
        "coverage": "missing",
        "effort": "high",
        "gaps": ["Section missing entirely"],
        "actions": ["Add Escalations section with severity ladder and approval matrix"]
      }
    ]
  },
  "section_strategy": {
    "verdict": "needs_improvement",
    "next_steps": [
      "Create explicit TOC",
      "Restructure into 9 H2 sections",
      "Add Governance section with ownership model",
      "Add Escalations section with severity ladder",
      "Replace passive voice with prescriptive language"
    ]
  }
}
```

**Expected Phase 2 Outputs:**
- 3/9 sections extracted (major gaps)
- 12 high-severity issues
- Overall posture: "needs_overhaul"

---

## **14. Implementation Checklist**

### **14.1 Phase 0: Foundation**

- [ ] Create `external/agent/doc_review_agent.py` skeleton
- [ ] Implement `AgentState` TypedDict in `external/doc_review/types.py`
- [ ] Create MCP tool wrappers:
  - [ ] `tool_load_document`
  - [ ] `tool_extract_headings`
  - [ ] `tool_detect_toc`
- [ ] Write unit tests for Phase 0 tools
- [ ] Validate with all 3 sample documents

### **14.2 Phase 1: Holistic Assessment**

- [ ] Implement LLM nodes:
  - [ ] `node_phase1_doc_summary_llm`
  - [ ] `node_phase1_toc_review_llm`
  - [ ] `node_phase1_template_fitness_llm`
  - [ ] `node_phase1_section_strategy_llm`
- [ ] Validate prompts against `policy_template`
- [ ] Write unit tests for each LLM node
- [ ] Run integration tests (Phase 1 E2E)
- [ ] Capture baseline outputs for regression testing

### **14.3 Phase 2: Section Review**

- [ ] Implement MCP tools:
  - [ ] `tool_extract_section_by_headings`
  - [ ] `tool_extract_section_by_toc`
- [ ] Implement LLM nodes:
  - [ ] `node_phase2_section_extraction_validator_llm`
  - [ ] `node_phase2_section_review_llm`
  - [ ] `node_phase2_summary_llm`
- [ ] Implement orchestrator helpers:
  - [ ] `orchestrator_phase2_extract_sections`
  - [ ] `orchestrator_phase2_section_reviews`
- [ ] Write unit tests for Phase 2 nodes
- [ ] Run integration tests (Phase 2 E2E)

### **14.4 Phase 3: Change Application**

- [ ] Implement MCP tool:
  - [ ] `tool_apply_changes_deterministic`
- [ ] Implement LLM nodes:
  - [ ] `node_change_selection_intent_llm`
  - [ ] `node_apply_changes_verifier_llm` (optional)
- [ ] Write unit tests for change application
- [ ] Run E2E tests (full workflow)

### **14.5 Orchestrator & WebSocket**

- [ ] Implement `orchestrate()` main loop
- [ ] Implement `run_node()` wrapper with WS events
- [ ] Create `NODE_REGISTRY` and `NODE_LABEL` mappings
- [ ] Test WebSocket event streaming

### **14.6 Autonomous Agent Layer**

- [ ] Implement `agent_planner_llm` node with fixed tool schema
  - [ ] Create prompt file `external/doc_review/prompts/agent_planner.md`
  - [ ] Define tool schema (run_phase1, run_phase2, apply_changes, etc.)
  - [ ] Implement JSON output validation
- [ ] Implement `handle_user_message(run_id, user_message)` entrypoint
  - [ ] Call `agent_planner_llm` to interpret user intent
  - [ ] Execute plan steps sequentially
  - [ ] Emit `agent_plan_generated` WebSocket event
- [ ] Implement phase wrapper functions:
  - [ ] `run_phase1(state)` – Calls orchestrator with `control="load_document"`
  - [ ] `run_phase2(state, section_scope=None)` – Calls orchestrator with `control="phase2_extract_sections"`
  - [ ] `run_phase3(state)` – Calls orchestrator with `control="apply_changes"`
  - [ ] `run_apply_changes(state, change_ids, severity_filter, sections)` – Filters changes and applies
- [ ] Implement VFS sync & locking functions:
  - [ ] `update_vfs(run_id, state)` – Sync AgentState to VFS
  - [ ] `emit_vfs_updates(state, paths)` – Emit `vfs_file_updated` events
  - [ ] `acquire_run_lock` / `release_run_lock` – Prevent concurrent plan execution
- [ ] Emit structured logs for each command (`logs/RUN_ID/agent_transcript.jsonl`)
- [ ] Write unit tests for agent planner (see 11.4)
- [ ] Test natural language commands E2E

### **14.7 VS Code Web Front-End**

- [ ] Bundle VS Code Web OSS with Flask server
  - [ ] Set up build environment (Node.js, yarn)
  - [ ] Configure custom branding (optional)
  - [ ] Serve static assets via Flask route `/doc-review`
  - [ ] Configure nginx reverse proxy + TLS
- [ ] Implement custom VFS provider (TypeScript):
  - [ ] `DocReviewVFS` class implementing `vscode.FileSystemProvider`
  - [ ] `stat(uri)` → GET `/vfs/stat?path={uri.path}`
  - [ ] `readFile(uri)` → GET `/vfs/file?path={uri.path}`
  - [ ] `readDirectory(uri)` → GET `/vfs/tree?path={uri.path}`
  - [ ] `writeFile(uri, content)` → PATCH `/vfs/file` (persist edits)
  - [ ] Register VFS provider with `vscode.workspace.registerFileSystemProvider`
- [ ] Implement backend VFS endpoints (Python):
  - [ ] `GET /vfs/stat` – Return file metadata (size, mtime, type)
  - [ ] `GET /vfs/file` – Return file content from AgentState
  - [ ] `GET /vfs/tree` – Return directory listing
  - [ ] `PATCH /vfs/file` – Update file content + emit `vfs_file_updated`
- [ ] Implement Agent Console (WebView panel):
  - [ ] Create WebView panel on right side
  - [ ] Implement chat UI (input + message history)
  - [ ] Implement execution trace (real-time stream)
  - [ ] Style with CSS (green/yellow/red for success/running/failed)
  - [ ] Make trace entries clickable (open artifact on click)
- [ ] Implement WebSocket listener for IDE events:
  - [ ] `vfs_file_updated` → Refresh file content + explorer tree
  - [ ] `open_artifact` → Open file in editor tab
  - [ ] `open_diff` → Trigger `vscode.diff` command
  - [ ] `agent_plan_generated` → Display plan in Agent Console
  - [ ] `node_started` / `node_completed` → Update execution trace
- [ ] Test VFS functionality:
  - [ ] Files appear in explorer
  - [ ] Clicking file opens in editor
  - [ ] JSON files render with syntax highlighting
  - [ ] Markdown files render with preview
- [ ] Test diff view:
  - [ ] `/versions/v1_original.md` ↔ `/versions/v2_after_changes.md`
  - [ ] Side-by-side diff renders correctly
  - [ ] Changes highlighted in green/red
- [ ] Add toolbar actions:
  - [ ] `Download latest version`
  - [ ] `Open README`

### **14.8 Testing & Validation**

- [ ] Run full test suite:
  - [ ] Unit tests (tools + nodes + agent planner)
  - [ ] Integration tests (phase-level)
  - [ ] E2E tests (full workflow + autonomous commands)
- [ ] Business validation with sample documents
- [ ] Test natural language commands:
  - [ ] "Run full review" → Phase 1 → Phase 2 → Phase 3
  - [ ] "Re-run Phase 2 for Governance only" → Only Governance re-processed
  - [ ] "Apply all high severity changes" → Correct subset applied, diff visible
  - [ ] "Show me the template fitness report" → File opens in editor
- [ ] Validate VFS + UI integration:
  - [ ] All artifacts visible and navigable via Explorer
  - [ ] Execution trace updates in real-time
  - [ ] Diff view works after apply changes
- [ ] Capture test results in `test/TEST_RESULTS_DOC_REVIEW.md`
- [ ] Update `docs/doc_review_steps.md` with final spec
- [ ] Ensure smoke scripts (Phase 1/2/3 + agent plan) run in CI

---

## **15. Success Metrics**

### **15.1 Technical Metrics**

- ✅ All unit tests pass (100% coverage for tools/nodes/agent planner)
- ✅ All integration tests pass (Phase 1, Phase 2, Phase 3)
- ✅ All E2E tests pass (3 sample documents + autonomous commands)
- ✅ WebSocket events stream correctly (no dropped events)
- ✅ Change application is deterministic (no LLM hallucinations)
- ✅ VFS syncs correctly with AgentState (all artifacts visible)
- ✅ Agent planner interprets natural language commands correctly (95%+ accuracy)

### **15.2 Business Metrics**

- ✅ Template_Version.pdf: verdict "ready", 0 high-severity issues
- ✅ Middle_Version.pdf: verdict "needs_improvement", 3 actionable steps
- ✅ Shitty_Version.pdf: verdict "needs_improvement", 5 actionable steps
- ✅ Template fitness detects concept coverage (not just heading matches)
- ✅ Suggested changes are specific and actionable (not abstract)
- ✅ Change application preserves surrounding text (no unintended edits)

### **15.3 User Experience Metrics**

- ✅ Natural language commands work intuitively ("Run full review", "Apply high severity")
- ✅ Execution trace updates in real-time (< 100ms latency)
- ✅ All artifacts navigable via VS Code Explorer
- ✅ Diff view renders correctly after applying changes
- ✅ Agent responses are clear and explain the plan
- ✅ Users can complete full workflow without reading documentation
- ✅ Manual edits persist across refreshes and sync to other sessions
- ✅ Download action retrieves the latest version with correct timestamp

---

**This specification is now production-ready for autonomous document review with VS Code Web interface. Begin implementation with Phase 0 foundation, then add agent layer and front-end.**

