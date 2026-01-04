# Frontend Holistic Review: System Prompts, Agent Instance Prompts, and Chatbot

## Executive Summary

**Overall Status**: ⚠️ **Partially Complete** - Core functionality exists but missing key features

**What Works**:
- ✅ System prompts management (admin UI)
- ✅ Generic chatbot interface
- ✅ Agent instance management

**What's Missing**:
- ❌ Per-agent prompt customization UI
- ❌ Evidence pack display for web research agents
- ❌ Prompt versioning/history UI
- ❌ Prompt testing interface

---

## 1. System Prompts Management

### Current Implementation

**Location**: `/admin` → "System Prompts" → "Structured" or "External"

**Features**:
- ✅ List prompts by category (`structured` or `web_search`)
- ✅ Edit prompt content (modal editor)
- ✅ Save prompt changes
- ✅ Display prompt metadata (name, description, filename)

**UI Components**:
```javascript
// Load prompts by category
loadPrompts('structured')  // or 'web_search'

// Edit prompt
editPrompt(promptName)

// Save prompt
savePrompt()
```

**API Endpoints Used**:
- `GET /api/admin/prompts?category=structured` - List prompts
- `GET /api/admin/prompts/<prompt_name>` - Get prompt content
- `POST /api/admin/prompts/<prompt_name>` - Save prompt content

**Code Location**: `web/templates/admin.html` lines 1082-1192

### What's Missing

1. **Prompt Versioning UI**
   - Backend has `PromptRevision` table
   - No UI to view/edit prompt history
   - No ability to revert to previous version

2. **Prompt Testing Interface**
   - No way to test prompts before saving
   - No preview of prompt with sample context
   - No validation feedback

3. **Prompt Comparison**
   - No diff view between versions
   - No side-by-side comparison

4. **Prompt Templates**
   - No template system for creating new prompts
   - No prompt library/examples

5. **Bulk Operations**
   - No export/import prompts
   - No bulk edit capabilities

---

## 2. Agent Instance Prompts (Per-Agent Customization)

### Current Backend Support

**Web Research Agents**: ✅ **Partially Supported**
- `load_web_research_decider_prompt(agent_id)` - Can load per-agent
- `load_web_research_synthesis_prompt(agent_id)` - Can load per-agent
- `get_prompt_content(db, prompt_name, category)` - Loads from DB
- **But**: No agent-specific prompt storage mechanism

**Structured Agents**: ❌ **Not Supported**
- `load_decider_prompt()` - No agent_id parameter
- Always loads from file or global DB prompt
- No per-agent customization

**Database Schema**:
```python
# Prompt table (global, not per-agent)
class Prompt:
    name: str  # Primary key
    category: str  # "structured" | "web_search"
    current_content: str
    # NO agent_id field - prompts are global
```

### What's Missing

1. **Per-Agent Prompt Override UI**
   - No UI to customize prompts per agent instance
   - No "Use default" vs "Override" toggle
   - No agent-specific prompt editor

2. **Agent Prompt Management**
   - No section in agent edit form for prompt overrides
   - No way to see which prompts are customized per agent
   - No inheritance visualization (default → override)

3. **Database Schema**
   - No `AgentPrompt` table for per-agent overrides
   - Would need: `agent_id`, `prompt_name`, `content`, `is_override`

4. **Backend Implementation**
   - Structured agents don't support `agent_id` in prompt loading
   - Need to add `agent_id` parameter to all prompt loaders

### Recommended Implementation

**Database Schema Addition**:
```python
class AgentPrompt(Base):
    __tablename__ = "agent_prompts"
    
    agent_id: Mapped[str] = ForeignKey("agents.id")
    prompt_name: Mapped[str] = ForeignKey("prompts.name")
    content: Mapped[str] = Text  # Override content
    is_active: Mapped[bool] = Boolean(default=True)
    
    # Composite primary key
    __table_args__ = (PrimaryKeyConstraint("agent_id", "prompt_name"),)
```

**Backend Changes**:
```python
# In decider.py
def load_decider_prompt(agent_id: Optional[str] = None) -> str:
    # 1. Try agent-specific override
    if agent_id:
        agent_prompt = get_agent_prompt(db, agent_id, "decider")
        if agent_prompt:
            return agent_prompt.content
    
    # 2. Try global DB prompt
    prompt = get_prompt_content(db, "decider", category="structured")
    if prompt:
        return prompt
    
    # 3. Fallback to file
    return load_from_file("decider.md")
```

**Frontend Changes**:
- Add "Prompts" tab in agent edit form
- Show list of prompts with "Override" toggle
- Allow editing agent-specific prompt content
- Show inheritance (default vs override)

---

## 3. Agent Chatbot Interface

### Current Implementation

**Location**: `/agent/chat/<agent_id>`

**Template**: `web/templates/agent_chat.html`

**Features**:
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
- ✅ SQL results table (for structured agents)

**SSE Events Handled**:
- `run_started`
- `decider_done`
- `sql_generated` (structured only)
- `results_ready` (structured only)
- `final_response`
- `ask_user`
- `run_blocked`
- `run_failed`
- `run_completed`

**Code Location**: `web/templates/agent_chat.html` lines 1239-1502

### What's Missing

1. **Evidence Pack Display** (Web Research Agents)
   - Backend sends `evidence_pack` in `final_response` event
   - Frontend doesn't parse/display it
   - Missing components:
     - Sources list with URLs
     - Claims display
     - Conflicts display
     - Gaps display

2. **Agent Type Detection in UI**
   - Chatbot doesn't detect agent type
   - Same UI for structured and web research
   - Should adapt based on agent type

3. **Response Formatting**
   - Structured: SQL + table ✅
   - Web Research: Text only (missing evidence) ❌

4. **Conversation Context**
   - No way to see prior query specs/research specs
   - No way to see agent's reasoning history
   - No way to export conversation

5. **Advanced Features**
   - No prompt injection testing
   - No conversation export
   - No conversation sharing
   - No conversation templates

---

## Detailed Gap Analysis

### System Prompts Management

| Feature | Status | Priority | Effort |
|---------|--------|----------|--------|
| List prompts by category | ✅ Complete | - | - |
| Edit prompt content | ✅ Complete | - | - |
| Save prompt changes | ✅ Complete | - | - |
| Prompt versioning/history | ❌ Missing | High | Medium |
| Prompt testing interface | ❌ Missing | Medium | High |
| Prompt comparison/diff | ❌ Missing | Low | Medium |
| Prompt templates | ❌ Missing | Low | Medium |
| Bulk export/import | ❌ Missing | Low | Low |

### Agent Instance Prompts

| Feature | Status | Priority | Effort |
|---------|--------|----------|--------|
| Backend support (web research) | ⚠️ Partial | - | - |
| Backend support (structured) | ❌ Missing | High | Medium |
| Database schema | ❌ Missing | High | Low |
| UI for prompt overrides | ❌ Missing | High | High |
| Agent prompt editor | ❌ Missing | High | Medium |
| Inheritance visualization | ❌ Missing | Medium | Medium |
| Default vs override toggle | ❌ Missing | High | Low |

### Chatbot Interface

| Feature | Status | Priority | Effort |
|---------|--------|----------|--------|
| Basic chat interface | ✅ Complete | - | - |
| SSE streaming | ✅ Complete | - | - |
| SQL results display | ✅ Complete | - | - |
| Evidence pack display | ❌ Missing | **Critical** | Medium |
| Sources list | ❌ Missing | **Critical** | Medium |
| Claims display | ❌ Missing | High | Medium |
| Conflicts display | ❌ Missing | High | Medium |
| Agent type detection | ❌ Missing | Medium | Low |
| Conversation export | ❌ Missing | Low | Low |
| Conversation sharing | ❌ Missing | Low | Medium |

---

## Implementation Roadmap

### Phase 1: Critical Missing Features (High Priority)

#### 1.1 Evidence Pack Display for Web Research Agents

**Files to Modify**:
- `web/templates/agent_chat.html`

**Changes Needed**:
```javascript
// Add after line 1360 (final_response handler)
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

**New Functions Needed**:
- `displayEvidencePack(messageEl, evidencePack)`
- `formatSourcesList(sources)`
- `formatClaimsList(claims)`
- `formatConflictsList(conflicts)`

**Estimated Effort**: 4-6 hours

#### 1.2 Per-Agent Prompt Override Backend

**Files to Modify**:
- `core/db/models.py` - Add `AgentPrompt` table
- `external/agent/decider.py` - Add `agent_id` parameter
- `external/agent/executor_nodes.py` - Add `agent_id` to prompt loaders
- `external/agent/persistence.py` - Add agent prompt CRUD functions

**Database Migration**:
```python
# alembic/versions/XXXX_add_agent_prompts.py
def upgrade():
    op.create_table(
        'agent_prompts',
        sa.Column('agent_id', sa.String(128), sa.ForeignKey('agents.id')),
        sa.Column('prompt_name', sa.String(128), sa.ForeignKey('prompts.name')),
        sa.Column('content', sa.Text()),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.PrimaryKeyConstraint('agent_id', 'prompt_name')
    )
```

**Estimated Effort**: 6-8 hours

### Phase 2: Agent Instance Prompts UI (High Priority)

#### 2.1 Agent Prompt Management UI

**Files to Create/Modify**:
- `web/templates/admin.html` - Add "Prompts" tab to agent edit form
- `routes/admin_routes.py` - Add agent prompt API endpoints

**New API Endpoints**:
- `GET /api/admin/agents/<agent_id>/prompts` - List agent prompts
- `GET /api/admin/agents/<agent_id>/prompts/<prompt_name>` - Get agent prompt
- `POST /api/admin/agents/<agent_id>/prompts/<prompt_name>` - Save agent prompt
- `DELETE /api/admin/agents/<agent_id>/prompts/<prompt_name>` - Delete override (use default)

**UI Components**:
- Prompt list with "Override" toggle
- Prompt editor (reuse existing modal)
- Inheritance indicator (shows "Using default" vs "Custom override")

**Estimated Effort**: 8-10 hours

### Phase 3: System Prompts Enhancements (Medium Priority)

#### 3.1 Prompt Versioning UI

**Files to Modify**:
- `web/templates/admin.html` - Add version history view
- `routes/admin_routes.py` - Add version API endpoints

**New Features**:
- View prompt revision history
- Compare versions (diff view)
- Revert to previous version

**Estimated Effort**: 6-8 hours

#### 3.2 Prompt Testing Interface

**Files to Create**:
- `web/templates/prompt_tester.html` - New page for testing prompts

**Features**:
- Load prompt template
- Fill in sample context
- Preview prompt with context
- Test LLM response (optional)

**Estimated Effort**: 8-10 hours

### Phase 4: Chatbot Enhancements (Low Priority)

#### 4.1 Agent Type Detection

**Files to Modify**:
- `web/templates/agent_chat.html` - Detect agent type on load
- Adapt UI based on agent type (show SQL section for structured, evidence section for web)

**Estimated Effort**: 2-3 hours

#### 4.2 Conversation Export

**Files to Modify**:
- `web/templates/agent_chat.html` - Add export button
- `routes/admin_routes.py` - Add export endpoint

**Formats**:
- Markdown
- JSON
- CSV (for structured results)

**Estimated Effort**: 4-6 hours

---

## Architecture Recommendations

### Prompt Loading Hierarchy

**Recommended Priority**:
1. **Agent-specific override** (if exists)
2. **Global DB prompt** (category-specific)
3. **File-based prompt** (fallback)

**Implementation**:
```python
def load_prompt_with_fallback(
    prompt_name: str,
    category: str,
    agent_id: Optional[str] = None
) -> str:
    # 1. Agent override
    if agent_id:
        override = get_agent_prompt(agent_id, prompt_name)
        if override and override.is_active:
            return override.content
    
    # 2. Global DB prompt
    global_prompt = get_prompt_content(prompt_name, category=category)
    if global_prompt:
        return global_prompt
    
    # 3. File fallback
    file_path = Path(f"external/config/prompts/{prompt_name}.md")
    if file_path.exists():
        return file_path.read_text()
    
    # 4. Error
    raise ValueError(f"Prompt {prompt_name} not found")
```

### UI Organization

**Recommended Structure**:
```
/admin
  ├── System Prompts
  │   ├── Structured (6 prompts)
  │   └── External (6 prompts)
  │       ├── Edit prompt
  │       ├── View history
  │       └── Test prompt
  │
  ├── Manage Agents
  │   ├── Structured
  │   │   └── [Agent] → Edit → Prompts tab
  │   └── External
  │       └── [Agent] → Edit → Prompts tab
  │
  └── Agent Chat
      └── /agent/chat/<agent_id>
          ├── Messages
          ├── Evidence Pack (web research)
          └── SQL Results (structured)
```

---

## Code Examples

### Evidence Pack Display Component

```javascript
function displayEvidencePack(messageEl, evidencePack) {
    if (!evidencePack) return;
    
    const sources = evidencePack.sources || [];
    const claims = evidencePack.claims || [];
    const conflicts = evidencePack.conflicts || [];
    const gaps = evidencePack.gaps || [];
    
    const evidenceDiv = document.createElement('details');
    evidenceDiv.className = 'evidence-pack mt-3';
    evidenceDiv.open = true;
    
    evidenceDiv.innerHTML = `
        <summary class="small text-info" style="cursor:pointer;">
            <i class="bi bi-file-earmark-text"></i> Research Evidence 
            (${sources.length} sources, ${claims.length} claims)
        </summary>
        <div class="evidence-content p-3 bg-light border rounded mt-2">
            ${formatSourcesList(sources)}
            ${formatClaimsList(claims)}
            ${formatConflictsList(conflicts)}
            ${formatGapsList(gaps)}
        </div>
    `;
    
    messageEl.appendChild(evidenceDiv);
}

function formatSourcesList(sources) {
    if (!sources || sources.length === 0) return '';
    
    let html = '<div class="sources-section mb-3"><h6>Sources</h6><ul class="list-unstyled">';
    sources.forEach((source, idx) => {
        html += `
            <li class="mb-2 p-2 border rounded">
                <a href="${escapeHtml(source.url)}" target="_blank" class="text-primary">
                    ${escapeHtml(source.title || source.url)}
                </a>
                ${source.snippet ? `<div class="small text-muted mt-1">${escapeHtml(source.snippet)}</div>` : ''}
            </li>
        `;
    });
    html += '</ul></div>';
    return html;
}
```

### Agent Prompt Override UI

```html
<!-- In agent edit form -->
<div class="form-tabs">
    <button class="tab-btn active" onclick="showTab('basic')">Basic</button>
    <button class="tab-btn" onclick="showTab('prompts')">Prompts</button>
    <button class="tab-btn" onclick="showTab('advanced')">Advanced</button>
</div>

<div id="prompts-tab" class="tab-content" style="display:none;">
    <h4>Prompt Overrides</h4>
    <p class="text-muted">Customize prompts for this agent instance. Leave empty to use system defaults.</p>
    
    <div id="agent-prompts-list">
        <!-- Loaded dynamically -->
    </div>
</div>
```

---

## Testing Checklist

### System Prompts
- [ ] List prompts by category
- [ ] Edit prompt content
- [ ] Save prompt changes
- [ ] View prompt history
- [ ] Revert to previous version
- [ ] Test prompt with sample context

### Agent Instance Prompts
- [ ] View agent prompts (default vs override)
- [ ] Create prompt override
- [ ] Edit agent-specific prompt
- [ ] Delete override (revert to default)
- [ ] Verify inheritance (default → override)

### Chatbot Interface
- [ ] Structured agent: SQL + table display
- [ ] Web research agent: Evidence pack display
- [ ] Sources list with clickable URLs
- [ ] Claims display with confidence
- [ ] Conflicts display with severity
- [ ] Agent type detection and UI adaptation

---

## Conclusion

**Current State**: 
- ✅ System prompts management (basic)
- ⚠️ Agent instance prompts (backend partial, UI missing)
- ⚠️ Chatbot interface (structured complete, web research incomplete)

**Critical Gaps**:
1. Evidence pack display for web research agents
2. Per-agent prompt customization UI
3. Prompt versioning/history UI

**Recommended Next Steps**:
1. Implement evidence pack display (Phase 1.1)
2. Add per-agent prompt backend support (Phase 1.2)
3. Build agent prompt management UI (Phase 2.1)

**Estimated Total Effort**: 30-40 hours for critical features

