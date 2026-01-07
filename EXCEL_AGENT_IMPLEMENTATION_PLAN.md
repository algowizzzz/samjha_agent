# Excel Agent Integration - Implementation Plan

## ✅ Architecture Alignment Confirmed

**Status:** Fully aligned with structured agent pattern
- System Prompt (data agnostic) → DB storage, editable in UI
- Domain File (data specific) → Uploaded, stored in DB
- Tools → Same MCP server, JSON configs
- Agent Type → Already registered ("excel")
- Route Integration → Same pattern as other agents

---

## 📋 Implementation Checklist

### Phase 1: Core Agent & Tools (Foundation)

#### 1.1 Create Fallback System Prompt File
**File:** `external/config/prompts/excel_agent_reasoning.md`
**Status:** ❌ Not Created
**Priority:** High
**Effort:** 30 min

**Content:** Data-agnostic Excel agent reasoning prompt (see EXCEL_AGENT_SYSTEM_PROMPTS.md for template)

**Action:**
```bash
# Create file with full reasoning prompt
# Include: Role, Workflow, Tool Usage, Planning Rules, Response Format
```

---

#### 1.2 Create Tool Configuration Files
**Files:** 
- `external/config/tools/list_excel_files.json`
- `external/config/tools/read_excel_sheet.json`
- `external/config/tools/query_excel_data.json`
**Status:** ❌ Not Created
**Priority:** High
**Effort:** 45 min

**Action:**
```json
// list_excel_files.json
{
  "name": "list_excel_files",
  "description": "List Excel files (.xlsx, .xls) in a data folder",
  "version": "1.0.0",
  "enabled": true,
  "class_path": "external.tools.excel_agent.list_excel_files.ListExcelFilesTool",
  "config": {
    "data_directory": "external/datawarehouse"
  }
}
```

**Verify:** ToolsRegistry auto-loads these on startup

---

#### 1.3 Create Unified Excel Base Agent
**File:** `external/agent/excel_base_agent.py`
**Status:** ❌ Not Created
**Priority:** High
**Effort:** 4-6 hours

**Key Functions:**
- `load_excel_reasoning_prompt(agent_id)` - Load from DB/file (same pattern as decider)
- `load_domain_md(agent_id)` - Load from agent record
- `_build_full_system_prompt()` - Combine reasoning + domain + config
- `handle_excel_query()` - Main entry point (ReAct loop)
- `_reason_initial_plan()` - LLM-based planning
- `_reason_next_plan()` - LLM-based next steps
- `_act_execute_tool()` - Execute tool calls
- `_observe_has_answer()` - Check completion
- `_reason_synthesize_answer()` - Final synthesis

**Pattern:** Merge `excel_agent.py` + `bmo_excel_agent.py` into unified agent

**Dependencies:**
- ToolsRegistry (for tool access)
- LLM client (for reasoning)
- Domain knowledge loading

---

#### 1.4 Create ECommerce Excel Domain File
**File:** `external/config/domains/ecommerce_excel_domain.md`
**Status:** ❌ Not Created
**Priority:** Medium
**Effort:** 1 hour

**Content:** Based on existing ECommerce structure, adapted for Excel format

**Action:**
- Review `external/config/domains/ecommerce_advanced_domain.md`
- Adapt for Excel file structure
- Document file relationships, column mappings, query patterns

---

### Phase 2: Database & Prompt Management

#### 2.1 Verify Prompt Storage Functions
**File:** `external/agent/persistence.py`
**Status:** ✅ Already Exists
**Priority:** Low
**Effort:** 15 min

**Action:**
- Verify `get_prompt_content()` supports `category="excel"`
- Test loading Excel prompt from DB
- Verify agent-specific override works

**Test:**
```python
from external.agent.persistence import get_prompt_content
with get_db_session() as db:
    prompt = get_prompt_content(db, "excel_agent_reasoning", category="excel")
```

---

#### 2.2 Create Initial System Prompt in DB
**Status:** ❌ Not Done
**Priority:** Medium
**Effort:** 30 min

**Action:**
- Use admin UI or direct SQL to insert default prompt
- Or: First agent creation will trigger fallback → file → can be edited in UI

**SQL:**
```sql
INSERT INTO prompts (id, name, category, agent_id, content, created_at, updated_at)
VALUES (
    gen_random_uuid()::text,
    'excel_agent_reasoning',
    'excel',
    NULL,  -- Global default
    '<content from excel_agent_reasoning.md>',
    NOW(),
    NOW()
);
```

---

### Phase 3: Route Integration

#### 3.1 Add Excel Case to Agent Run Routes
**File:** `external/routes/agent_run_routes.py`
**Status:** ❌ Not Done
**Priority:** High
**Effort:** 1 hour

**Location:** Around line 633 (after quick_search, before structured default)

**Code:**
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

**Also Update:**
- Event emission (if needed)
- Error handling
- Result formatting (match expected response structure)

---

#### 3.2 Test Route Integration
**Status:** ❌ Not Done
**Priority:** High
**Effort:** 30 min

**Action:**
- Create test Excel agent via API
- Execute test query
- Verify response format matches UI expectations

---

### Phase 4: UI Integration

#### 4.1 Add System Prompts → Excel Section
**File:** `external/web/templates/admin.html`
**Status:** ❌ Not Done
**Priority:** Medium
**Effort:** 2 hours

**Action:**
1. Add sidebar item:
```html
<div class="sidebar-item" data-section="system-prompts-excel" onclick="showContent('system-prompts-excel')">
    System Prompts → Excel
</div>
```

2. Add content section (copy pattern from "system-prompts-structured"):
```html
<div class="content-section" id="content-system-prompts-excel">
    <div class="content-header">
        <h2>📊 System Prompts - Excel Agents</h2>
        <p>Edit system prompts for Excel data agents</p>
    </div>
    <!-- Prompt editor (same as structured) -->
</div>
```

3. Add JavaScript handlers (copy from structured section)

---

#### 4.2 Add Manage Agents → Excel Section
**File:** `external/web/templates/admin.html`
**Status:** ❌ Not Done
**Priority:** High
**Effort:** 4-6 hours

**Action:**
1. Add sidebar item:
```html
<div class="sidebar-item" data-section="manage-agents-excel" onclick="showContent('manage-agents-excel')">
    Manage Agents → Excel
</div>
```

2. Add content section with:
   - Agent list display
   - Create/Edit form
   - Form fields:
     - Name, Description
     - Domain File upload (required)
     - Data Folder selection
     - Excel-specific config:
       - Read All Sheets (checkbox)
       - Max Iterations (number)
       - Max Rows Per Sheet (number)
     - Excel Files upload (optional, multiple)

3. Add JavaScript:
   - `showCreateAgentForm('excel')`
   - `saveAgent('excel')`
   - `loadAgents('excel')`
   - Form validation
   - File upload handling

**Pattern:** Copy from "manage-agents-structured" and adapt

---

#### 4.3 Update Agent Selection Page
**File:** `external/web/templates/agents.html`
**Status:** ❌ Not Done
**Priority:** Medium
**Effort:** 1 hour

**Action:**
- Ensure Excel agents appear in agent selection
- Group by type (Excel agents separate section)
- Display agent info (name, description, data folder)

---

#### 4.4 Update Agent Chat UI (if needed)
**File:** `external/web/templates/agent_chat.html`
**Status:** ⚠️ May Need Updates
**Priority:** Low
**Effort:** 1-2 hours

**Action:**
- Verify Excel agent responses display correctly
- Add Excel-specific features (if desired):
  - Show sheets read
  - Display file structure
  - Export to Excel

---

### Phase 5: Backend API Updates

#### 5.1 Update Agent Creation API
**File:** `external/routes/agent_routes.py` (or similar)
**Status:** ❌ Not Verified
**Priority:** High
**Effort:** 2 hours

**Action:**
- Verify agent creation endpoint handles Excel agent type
- Ensure domain file upload works
- Store Excel-specific config (read_all_sheets, etc.)
- Save domain_content to DB

**Test:**
```python
POST /api/agents
{
    "name": "BMO Financials Agent",
    "description": "...",
    "agent_type": "excel",
    "domain_file": <file>,
    "data_folder": "bmo_financials",
    "config": {
        "read_all_sheets": true,
        "max_iterations": 15,
        "max_rows_per_sheet": 100
    }
}
```

---

#### 5.2 Update Agent List API
**Status:** ⚠️ May Need Updates
**Priority:** Low
**Effort:** 30 min

**Action:**
- Verify Excel agents appear in agent list
- Filter by agent_type="excel" works

---

### Phase 6: Testing & Validation

#### 6.1 Unit Tests
**Status:** ❌ Not Created
**Priority:** Medium
**Effort:** 2-3 hours

**Tests:**
- Prompt loading (DB → file fallback)
- Domain file loading
- Tool execution
- ReAct loop logic
- Response formatting

---

#### 6.2 Integration Tests
**Status:** ❌ Not Created
**Priority:** High
**Effort:** 3-4 hours

**Tests:**
1. **BMO Agent:**
   - Create agent through UI
   - Upload domain file
   - Execute query: "List all files"
   - Execute query: "Revenue breakdown by segment Q4 2024"
   - Verify all sheets read (if read_all_sheets=true)

2. **ECommerce Agent:**
   - Create agent through UI
   - Upload domain file
   - Execute query: "Show sales data"
   - Execute query: "Top customers by revenue"
   - Verify correct file navigation

3. **System Prompt Editing:**
   - Edit global Excel prompt in UI
   - Create agent with override
   - Verify override works

---

#### 6.3 End-to-End Testing
**Status:** ❌ Not Done
**Priority:** High
**Effort:** 2-3 hours

**Scenarios:**
1. Create BMO agent → Query → Verify results
2. Create ECommerce agent → Query → Verify results
3. Edit system prompt → Verify changes apply
4. Edit domain file → Verify changes apply
5. Multiple agents → Verify isolation
6. Error handling → Verify graceful failures

---

### Phase 7: Documentation & Cleanup

#### 7.1 Update Documentation
**Status:** ⚠️ Partial
**Priority:** Low
**Effort:** 1 hour

**Files:**
- README.md (if exists)
- Agent creation guide
- Domain file template

---

#### 7.2 Code Cleanup
**Status:** ❌ Not Done
**Priority:** Low
**Effort:** 1 hour

**Action:**
- Remove old `bmo_excel_agent.py` (if keeping unified version)
- Remove test files (or move to tests/)
- Add docstrings
- Add type hints

---

## 📊 Implementation Order (Recommended)

### Week 1: Foundation
1. ✅ Create tool configs (1.2) - **30 min**
2. ✅ Create fallback prompt file (1.1) - **30 min**
3. ✅ Create unified Excel agent (1.3) - **4-6 hours**
4. ✅ Test tool loading - **30 min**

### Week 1: Integration
5. ✅ Add route integration (3.1) - **1 hour**
6. ✅ Test route integration (3.2) - **30 min**
7. ✅ Create ECommerce domain (1.4) - **1 hour**

### Week 2: UI
8. ✅ Add System Prompts UI (4.1) - **2 hours**
9. ✅ Add Manage Agents UI (4.2) - **4-6 hours**
10. ✅ Update agent selection (4.3) - **1 hour**

### Week 2: Testing
11. ✅ Integration tests (6.2) - **3-4 hours**
12. ✅ End-to-end tests (6.3) - **2-3 hours**
13. ✅ Fix issues found - **Variable**

### Week 3: Polish
14. ✅ Documentation (7.1) - **1 hour**
15. ✅ Code cleanup (7.2) - **1 hour**
16. ✅ Final validation - **2 hours**

---

## 🔍 Critical Dependencies

### Must Have Before Starting:
- ✅ Excel tools created (`external/tools/excel_agent/`)
- ✅ Agent type registered (`"excel"` in agent_registry)
- ✅ ToolsRegistry pattern understood
- ✅ Prompt storage pattern understood

### Nice to Have:
- ECommerce domain file (can create during implementation)
- Test data (BMO and ECommerce Excel files)

---

## ⚠️ Potential Issues & Solutions

### Issue 1: Tool Loading
**Problem:** Tools not auto-loaded by ToolsRegistry
**Solution:** Verify JSON configs are in `external/config/tools/` and follow correct format

### Issue 2: Prompt Loading
**Problem:** Prompt not found
**Solution:** Ensure fallback file exists, DB entry created, or first use creates it

### Issue 3: Domain File Not Loading
**Problem:** Domain content empty
**Solution:** Verify file upload works, content saved to `domain_content` field

### Issue 4: Route Not Routing
**Problem:** Excel agent not executing
**Solution:** Verify agent_type="excel" in DB, route case added correctly

### Issue 5: UI Not Showing
**Problem:** Excel sections not visible
**Solution:** Verify JavaScript handlers, section IDs match, CSS classes correct

---

## 📝 Success Criteria

### Phase 1 Complete When:
- [ ] Tool configs created and tools load successfully
- [ ] Fallback prompt file exists
- [ ] Unified Excel agent created and imports work
- [ ] Can call `handle_excel_query()` directly

### Phase 2 Complete When:
- [ ] Prompt loads from DB (or file fallback)
- [ ] Domain file loads from agent record
- [ ] Full prompt assembly works

### Phase 3 Complete When:
- [ ] Route integration added
- [ ] Test query executes through API
- [ ] Response format matches expectations

### Phase 4 Complete When:
- [ ] System Prompts → Excel section visible and editable
- [ ] Manage Agents → Excel section visible
- [ ] Can create Excel agent through UI
- [ ] Domain file uploads work
- [ ] Excel-specific config saves

### Phase 5 Complete When:
- [ ] Agent creation API works
- [ ] Agent list API includes Excel agents
- [ ] All CRUD operations work

### Phase 6 Complete When:
- [ ] BMO agent works end-to-end
- [ ] ECommerce agent works end-to-end
- [ ] System prompt editing works
- [ ] Domain file editing works
- [ ] Error handling graceful

### Phase 7 Complete When:
- [ ] Documentation updated
- [ ] Code cleaned up
- [ ] All tests pass
- [ ] Ready for production

---

## 🎯 Estimated Total Effort

**Minimum:** 25-30 hours
**Realistic:** 35-40 hours
**With Buffer:** 40-50 hours

**Breakdown:**
- Core Agent: 6-8 hours
- Route Integration: 2 hours
- UI Integration: 8-10 hours
- Testing: 8-10 hours
- Documentation: 2 hours
- Buffer: 5-10 hours

---

## 🚀 Quick Start (Minimal Viable Integration)

**If time-constrained, do this first:**

1. ✅ Create tool configs (1.2) - **30 min**
2. ✅ Create unified agent (1.3) - **4-6 hours** (simplified version)
3. ✅ Add route integration (3.1) - **1 hour**
4. ✅ Test via API directly - **30 min**

**This gets Excel agent working via API. UI can come later.**

---

**Status:** Plan ready. All angles reviewed. Architecture confirmed aligned.

**Next Step:** Start with Phase 1.1 (Create fallback prompt file) or Phase 1.2 (Create tool configs) - whichever is faster to validate the pattern.

