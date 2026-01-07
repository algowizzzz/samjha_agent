# Excel Agent Architecture - Aligned with Structured Agent

## Overview

Excel Agent follows the **same architecture pattern** as Structured Agent:
- **System Prompt** (data agnostic) - stored in DB, editable in UI
- **Domain File** (data specific) - uploaded when creating agent, stored in DB
- Both are **appended together** to form the full system prompt

---

## 1. System Prompt (Data Agnostic)

### A) Storage

**Database Table:** `prompts` (same as structured agent)

**Fields:**
- `name`: `"excel_agent_reasoning"`
- `category`: `"excel"`
- `agent_id`: Optional (for agent-specific overrides)
- `content`: The prompt text

### B) Loading Pattern

**File:** `external/agent/excel_base_agent.py`

```python
def load_excel_reasoning_prompt(agent_id: Optional[str] = None) -> str:
    """Load Excel agent reasoning prompt from DB (with agent override) or file."""
    # Try to load from DB first (with agent override if provided)
    if agent_id:
        try:
            from external.core.db.session import get_db_session
            from external.agent.persistence import get_prompt_content
            with get_db_session() as db:
                prompt_content = get_prompt_content(db, "excel_agent_reasoning", category="excel", agent_id=agent_id)
                if prompt_content:
                    return prompt_content
        except Exception as e:
            logger.warning(f"Failed to load prompt from DB for agent {agent_id}: {e}")
    
    # Try global DB prompt
    try:
        from external.core.db.session import get_db_session
        from external.agent.persistence import get_prompt_content
        with get_db_session() as db:
            prompt_content = get_prompt_content(db, "excel_agent_reasoning", category="excel")
            if prompt_content:
                return prompt_content
    except Exception as e:
        logger.warning(f"Failed to load prompt from DB: {e}")
    
    # Fallback to file
    prompt_path = Path("external/config/prompts/excel_agent_reasoning.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        logger.warning(f"Excel reasoning prompt not found at {prompt_path}, using fallback")
        return "# EXCEL AGENT REASONING PROMPT\n\nOutput JSON only."
```

### C) UI Location

**Admin UI:** `System Prompts → Excel`

- Editable text area
- Save to DB with `category="excel"`, `name="excel_agent_reasoning"`
- Can have global default + agent-specific overrides

---

## 2. Domain File (Data Specific)

### A) Storage

**Database Table:** `agents` (same as structured agent)

**Field:** `domain_content` (text/blob)

**Also stored as:** File reference in `domain_file` field (filename)

### B) Loading Pattern

**File:** `external/agent/excel_base_agent.py`

```python
def load_domain_md(agent_id: Optional[str] = None) -> str:
    """Load domain markdown content from database."""
    if agent_id:
        try:
            from external.core.db.session import get_db_session
            from external.agent.persistence import get_agent_db
            
            with get_db_session() as db:
                agent = get_agent_db(db, agent_id)
                if agent:
                    domain_content = agent.get("domain_content")
                    if domain_content:
                        return domain_content
                    else:
                        logger.warning(f"Domain content not found for agent {agent_id}")
        except Exception as e:
            logger.warning(f"Error loading domain content: {e}")
    
    return ""
```

### C) UI Location

**Admin UI:** `Manage Agents → Excel → Create/Edit Agent Form`

- File upload field: `<input type="file" name="domain_file" accept=".md,.txt">`
- Uploaded when creating/editing agent
- Stored in DB as `domain_content`
- Also saved to `external/config/domains/{agent_id}_domain.md` (for reference)

---

## 3. Full Prompt Assembly

### A) How They're Combined

**File:** `external/agent/excel_base_agent.py`

```python
def _build_full_system_prompt(
    self,
    user_query: str,
    conversation_history: List[Dict],
    observations: List[Dict],
    config: Dict
) -> str:
    """Build full system prompt by combining reasoning prompt + domain knowledge."""
    
    # 1. Load system prompt (data agnostic)
    reasoning_prompt = load_excel_reasoning_prompt(agent_id=self.agent_id)
    
    # 2. Load domain file (data specific)
    domain_md = load_domain_md(agent_id=self.agent_id)
    
    # 3. Assemble full prompt
    full_prompt = f"""{reasoning_prompt}

---

## DOMAIN KNOWLEDGE

{domain_md}

---

## CONFIGURATION

{json.dumps(config, indent=2)}

---

## USER QUERY

{user_query}

---

## CONVERSATION HISTORY

{json.dumps(conversation_history, indent=2)}

---

## OBSERVATIONS

{json.dumps(observations, indent=2)}
"""
    
    return full_prompt
```

### B) Usage in LLM Call

```python
# Build full prompt
system_prompt = self._build_full_system_prompt(...)

# Call LLM
response = llm_client.invoke_with_prompt(
    system_prompt=system_prompt,  # Full prompt (reasoning + domain)
    user_prompt="Plan your next actions based on the context above."
)
```

---

## 4. UI Structure (Same as Structured Agent)

### A) System Prompts Section

**Location:** `external/web/templates/admin.html`

**Add:**
```html
<div class="sidebar-item" data-section="system-prompts-excel" onclick="showContent('system-prompts-excel')">
    System Prompts → Excel
</div>
```

**Content Section:**
```html
<div class="content-section" id="content-system-prompts-excel">
    <div class="content-header">
        <h2>📊 System Prompts - Excel Agents</h2>
        <p>Edit system prompts for Excel data agents</p>
    </div>
    <!-- Prompt editor (same pattern as structured) -->
</div>
```

### B) Manage Agents Section

**Location:** `external/web/templates/admin.html`

**Add:**
```html
<div class="sidebar-item" data-section="manage-agents-excel" onclick="showContent('manage-agents-excel')">
    Manage Agents → Excel
</div>
```

**Content Section:**
```html
<div class="content-section" id="content-manage-agents-excel">
    <div class="content-header">
        <h2>📊 Manage Agents - Excel</h2>
        <button class="btn btn-primary" onclick="showCreateAgentForm('excel')">+ Create New Agent</button>
    </div>
    
    <!-- Create/Edit Agent Form -->
    <form id="create-agent-form-excel">
        <input type="hidden" name="agent_type" value="excel">
        
        <!-- Basic Info -->
        <div class="form-group">
            <label>Agent Name *</label>
            <input type="text" name="name" required>
        </div>
        
        <div class="form-group">
            <label>Description</label>
            <textarea name="description"></textarea>
        </div>
        
        <!-- System Prompt (optional override) -->
        <div class="form-group">
            <label>System Prompt Override (Optional)</label>
            <textarea name="system_prompt_override"></textarea>
            <small>Leave empty to use global Excel agent reasoning prompt</small>
        </div>
        
        <!-- Domain File (REQUIRED) -->
        <div class="form-group">
            <label>Domain File *</label>
            <input type="file" name="domain_file" accept=".md,.txt" required>
            <small>Upload domain-specific knowledge (.md or .txt)</small>
        </div>
        
        <!-- Data Folder -->
        <div class="form-group">
            <label>Data Folder *</label>
            <select name="data_folder">
                <!-- List of folders in datawarehouse -->
            </select>
        </div>
        
        <!-- Excel-Specific Config -->
        <div class="form-group">
            <label>Read All Sheets</label>
            <input type="checkbox" name="read_all_sheets" checked>
            <small>If checked, reads all sheets from Excel files (for comprehensive analysis)</small>
        </div>
        
        <div class="form-group">
            <label>Max Iterations</label>
            <input type="number" name="max_iterations" value="15" min="1" max="50">
        </div>
        
        <div class="form-group">
            <label>Max Rows Per Sheet</label>
            <input type="number" name="max_rows_per_sheet" value="100" min="10" max="1000">
        </div>
        
        <!-- Excel Files Upload (Optional) -->
        <div class="form-group">
            <label>Excel Files (Optional)</label>
            <input type="file" name="excel_files" multiple accept=".xlsx,.xls">
            <small>Upload Excel files to the data folder</small>
        </div>
        
        <button type="submit">Create Agent</button>
    </form>
</div>
```

---

## 5. Database Schema

### A) Prompts Table

```sql
CREATE TABLE prompts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,  -- "excel_agent_reasoning"
    category TEXT NOT NULL,  -- "excel"
    agent_id TEXT,  -- NULL for global, agent_id for override
    content TEXT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

**Example Rows:**
- Global: `name="excel_agent_reasoning"`, `category="excel"`, `agent_id=NULL`
- Agent Override: `name="excel_agent_reasoning"`, `category="excel"`, `agent_id="bmo_financials_agent"`

### B) Agents Table

```sql
CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    agent_type TEXT NOT NULL,  -- "excel"
    domain_file TEXT,  -- Filename reference
    domain_content TEXT,  -- Actual domain markdown content
    data_folder TEXT,
    config JSON,  -- {"read_all_sheets": true, "max_iterations": 15, ...}
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## 6. Tools Registration

### A) Tools Location

**Same MCP Server** as existing tools:
- `external/tools/excel_agent/list_excel_files.py`
- `external/tools/excel_agent/read_excel_sheet.py`
- `external/tools/excel_agent/query_excel_data.py`

### B) Tool Configs

**Location:** `external/config/tools/`

- `list_excel_files.json`
- `read_excel_sheet.json`
- `query_excel_data.json`

**Registered via:** `ToolsRegistry` (same as other tools)

---

## 7. Agent Type Registration

### A) Already Done ✅

**File:** `external/agent/agent_registry.py`

```python
def validate_agent_type(agent_type: str) -> None:
    if agent_type not in ("structured", "unstructured", "external", "excel", "deep_research", "quick_search"):
        raise ValueError("Invalid agent_type")
```

**Status:** `"excel"` is already registered as valid agent type.

---

## 8. Route Integration

### A) Agent Run Routes

**File:** `external/routes/agent_run_routes.py`

**Add Excel case (around line 633):**

```python
elif agent_type == "excel":
    # Excel agent
    from external.agent.excel_base_agent import handle_excel_query
    result = handle_excel_query(
        user_query=user_query,
        agent_id=agent_id,
        conversation_history=conversation_history,
        prior_state=None,  # Excel agent manages its own state
        show_thinking=show_thinking,
    )
```

---

## 9. Summary: Architecture Alignment

| Component | Structured Agent | Excel Agent | Status |
|-----------|------------------|-------------|--------|
| **System Prompt** | `decider.md` (category="structured") | `excel_agent_reasoning.md` (category="excel") | ✅ Same pattern |
| **Domain File** | Uploaded, stored in `domain_content` | Uploaded, stored in `domain_content` | ✅ Same pattern |
| **Prompt Assembly** | System prompt + domain_md | System prompt + domain_md | ✅ Same pattern |
| **UI Location** | System Prompts → Structured | System Prompts → Excel | ⚠️ Need to add |
| **UI Location** | Manage Agents → Structured | Manage Agents → Excel | ⚠️ Need to add |
| **Tools** | Same MCP server | Same MCP server | ✅ Same pattern |
| **Agent Type** | Registered in agent_registry | Registered in agent_registry | ✅ Already done |
| **Route Integration** | In agent_run_routes.py | In agent_run_routes.py | ⚠️ Need to add |

---

## 10. Implementation Checklist

### Phase 1: Core Files
- [x] Excel tools created
- [ ] Create `excel_agent_reasoning.md` (fallback prompt file)
- [ ] Create `excel_base_agent.py` (unified agent)
- [ ] Create tool JSON configs (3 files)

### Phase 2: Database & Loading
- [ ] Implement `load_excel_reasoning_prompt()` (same pattern as decider)
- [ ] Implement `load_domain_md()` for Excel agent (same pattern as structured)
- [ ] Test prompt loading (DB → file fallback)

### Phase 3: UI Integration
- [ ] Add "System Prompts → Excel" section in admin.html
- [ ] Add "Manage Agents → Excel" section in admin.html
- [ ] Add Excel agent creation form (with domain file upload)
- [ ] Add Excel-specific config fields (read_all_sheets, max_iterations, etc.)

### Phase 4: Route Integration
- [ ] Add Excel case in `agent_run_routes.py`
- [ ] Test agent execution through API

### Phase 5: Testing
- [ ] Create BMO agent through UI
- [ ] Create ECommerce agent through UI
- [ ] Test system prompt editing
- [ ] Test domain file upload
- [ ] Test query execution

---

**Status:** Architecture is clear and aligned with structured agent pattern. Ready for implementation.

