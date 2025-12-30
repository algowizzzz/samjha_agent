# Web Search Agent: System Prompts Inventory

## Current State

### Structured Agent Prompts (Existing)
Located in: `external/config/prompts/`

1. **decider.md** - Gate prompt (creates QuerySpec)
2. **ask_user_clarification.md** - User clarification questions
3. **nl_to_sql_planner.md** - SQL generation (Executor)
4. **query_result_evaluator.md** - Result evaluation (Executor)
5. **response_commentary.md** - Final response formatting
6. **sql_plan_updater.md** - SQL patching (Executor)

**Admin Panel:** ✅ Already supports editing these (category: "structured")

---

## Web Search Agent Prompts Needed

### Required Prompts (NEW)

| Prompt Name | Category | Purpose | Replaces |
|-------------|----------|---------|----------|
| **web_research_decider.md** | `web_search` | Gate: Creates ResearchSpec, decides ASK_USER/EXECUTE/BLOCK | `decider.md` (structured) |
| **web_research_synthesis.md** | `web_search` | Final answer generation from EvidencePack | `query_result_evaluator.md` + `response_commentary.md` |
| **web_research_claim_extraction.md** | `web_search` | Extract claims from source content (Executor) | N/A (new) |
| **web_research_conflict_detection.md** | `web_search` | Detect conflicts between claims (Executor) | N/A (new) |

### Reusable Prompts (SHARED)

| Prompt Name | Category | Purpose | Used By |
|-------------|----------|---------|---------|
| **ask_user_clarification.md** | `shared` | User clarification questions | Both structured & web search |
| **response_commentary.md** | `shared` | Natural language response formatting | Both structured & web search |

---

## Total Prompt Count

### By Category

- **Structured Agents:** 6 prompts
- **Web Search Agents:** 4 prompts (2 new + 2 shared)
- **Shared:** 2 prompts
- **Total Unique:** 8 prompts

### By Function

- **Gate/Decider:** 2 (decider.md, web_research_decider.md)
- **Executor:** 4 (nl_to_sql_planner.md, query_result_evaluator.md, sql_plan_updater.md, web_research_claim_extraction.md, web_research_conflict_detection.md)
- **Synthesis:** 2 (response_commentary.md, web_research_synthesis.md)
- **User Interaction:** 1 (ask_user_clarification.md)

---

## Admin Panel Integration

### Current Admin Panel Support

**✅ Already Implemented:**
- Prompt listing by category (`/api/admin/prompts?category=structured`)
- Prompt editing modal
- Prompt saving (`/api/admin/prompts/<name>` POST)
- Category filtering (structured, unstructured, web-based)

**Location in UI:**
- `/admin` → "System Prompts" section
- Currently shows "Structured Prompts" and placeholder for "Web-based Prompts"

### What Needs to Be Added

1. **Create new prompts** for web search category
2. **Update admin panel** to show web search prompts
3. **Add prompt creation** (currently only editing exists)
4. **Category management** (ensure "web_search" category works)

---

## Prompt Details

### 1. web_research_decider.md

**Category:** `web_search`  
**Purpose:** Gate prompt - Creates ResearchSpec, decides action  
**Replaces:** `decider.md` (structured agent version)

**Key Sections:**
- Role: Decider (Gate) for web research
- Inputs: user_query, domain_content, prior_research_spec, evidence_pack
- Output: ResearchSpec + Plan
- Decision rubric: ASK_USER / EXECUTE / BLOCK
- ResearchSpec schema
- Research plan structure

**Size:** ~500-800 lines (similar to structured decider.md)

---

### 2. web_research_synthesis.md

**Category:** `web_search`  
**Purpose:** Generate final answer from EvidencePack  
**Replaces:** `query_result_evaluator.md` + `response_commentary.md` (combined)

**Key Sections:**
- Role: Synthesize evidence into final answer
- Inputs: evidence_pack, research_spec, user_query
- Output: Final report with citations
- Conflict handling
- Confidence levels
- Citation formatting

**Size:** ~200-300 lines

---

### 3. web_research_claim_extraction.md

**Category:** `web_search`  
**Purpose:** Extract claims from source content (Executor node)  
**New:** Not in structured agents

**Key Sections:**
- Role: Extract factual claims from source text
- Input: Source content, research_spec scope
- Output: List of claims with confidence
- Claim format: claim_text, supported_by, confidence

**Size:** ~150-200 lines

---

### 4. web_research_conflict_detection.md

**Category:** `web_search`  
**Purpose:** Detect conflicts between claims (Executor node)  
**New:** Not in structured agents

**Key Sections:**
- Role: Detect contradictory claims
- Input: List of claims
- Output: Conflicts list with severity
- Conflict types: direct negation, quantitative disagreement, temporal disagreement

**Size:** ~150-200 lines

---

## Admin Panel UI Updates Needed

### A. System Prompts Section

**Current:**
```
System Prompts
├── Structured Prompts (✅ working)
├── Unstructured Prompts (placeholder)
└── Web-based Prompts (placeholder)
```

**Updated:**
```
System Prompts
├── Structured Prompts (✅ working)
│   ├── decider.md
│   ├── nl_to_sql_planner.md
│   ├── query_result_evaluator.md
│   ├── sql_plan_updater.md
│   ├── ask_user_clarification.md
│   └── response_commentary.md
├── Unstructured Prompts (placeholder)
└── Web Search Prompts (✅ NEW)
    ├── web_research_decider.md
    ├── web_research_synthesis.md
    ├── web_research_claim_extraction.md
    ├── web_research_conflict_detection.md
    ├── ask_user_clarification.md (shared)
    └── response_commentary.md (shared)
```

### B. Prompt Creation (Missing Feature)

**Current:** Only editing existing prompts  
**Needed:** Create new prompts

**UI Addition:**
```
[+ Create New Prompt] button
     ↓
Modal:
- Prompt Name: [web_research_decider]
- Category: [web_search ▼]
- Template: [Blank | Copy from existing]
- Content: [Textarea]
```

### C. Category Management

**Current Categories:**
- `structured` (working)
- `unstructured` (placeholder)
- `web_search` (needs to be added)

**Database:** Prompts table has `category` column (already supports this)

---

## Implementation Checklist

### Phase 1: Create Prompts
- [ ] Create `web_research_decider.md` (based on structured decider.md)
- [ ] Create `web_research_synthesis.md` (new)
- [ ] Create `web_research_claim_extraction.md` (new)
- [ ] Create `web_research_conflict_detection.md` (new)
- [ ] Mark `ask_user_clarification.md` as `shared` category
- [ ] Mark `response_commentary.md` as `shared` category

### Phase 2: Admin Panel Updates
- [ ] Update `loadPrompts()` to support `web_search` category
- [ ] Add "Web Search Prompts" section in admin.html
- [ ] Add prompt creation UI (currently missing)
- [ ] Update prompt editor to show category
- [ ] Add category filter dropdown

### Phase 3: Backend Updates
- [ ] Ensure `category` field supports `web_search`
- [ ] Add prompt creation endpoint (if missing)
- [ ] Update prompt loading logic to support categories
- [ ] Add validation for prompt names/categories

### Phase 4: Integration
- [ ] Update `web_research_decider.py` to load prompt from DB
- [ ] Update `web_research_executor_nodes.py` to load prompts
- [ ] Update `web_research_synthesis.py` to load prompt
- [ ] Test prompt editing in admin panel
- [ ] Test prompt loading at runtime

---

## Prompt Loading Logic

### Current (Structured Agents)

```python
# In decider.py
def load_decider_prompt() -> str:
    prompt_path = Path("external/config/prompts/decider.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        # Fallback: load from DB
        from core.db.session import get_db_session
        from external.agent.persistence import get_prompt_content
        with get_db_session() as db:
            content = get_prompt_content(db, "decider")
            if content:
                return content
        return "# DECIDER PROMPT\n\nOutput JSON only."
```

### Updated (Web Search Agents)

```python
# In web_research_decider.py
def load_web_research_decider_prompt() -> str:
    # Try DB first (admin panel edits)
    from core.db.session import get_db_session
    from external.agent.persistence import get_prompt_content
    with get_db_session() as db:
        content = get_prompt_content(db, "web_research_decider")
        if content:
            return content
    
    # Fallback: file system
    prompt_path = Path("external/config/prompts/web_research_decider.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    
    # Final fallback
    return "# WEB RESEARCH DECIDER PROMPT\n\nOutput JSON only."
```

---

## Database Schema

### Prompts Table (Already Exists)

```sql
CREATE TABLE prompts (
    name VARCHAR(128) PRIMARY KEY,
    category VARCHAR(32) NOT NULL DEFAULT 'structured',
    current_content TEXT NOT NULL,
    updated_at DATETIME NOT NULL
);
```

**Categories:**
- `structured` (existing)
- `unstructured` (future)
- `web_search` (new)
- `shared` (new - for reusable prompts)

---

## Summary

### Prompt Count

**Total:** 8 unique prompts
- 6 for structured agents (existing)
- 4 for web search agents (2 new + 2 shared)
- 2 shared (used by both)

### Admin Panel Status

**✅ Already Working:**
- Prompt listing
- Prompt editing
- Category filtering
- Database storage

**❌ Missing:**
- Prompt creation UI
- Web search category display
- Shared category support

### Next Steps

1. **Create the 4 new prompts** (web_research_*.md)
2. **Update admin panel** to show web search prompts
3. **Add prompt creation** feature
4. **Test prompt loading** at runtime

---

## Key Insight

**All system prompts should be editable via admin panel** - this gives you:
- ✅ No code changes needed to tweak prompts
- ✅ A/B testing different prompt versions
- ✅ Domain-specific prompt customization
- ✅ Version control via PromptRevision table

The admin panel already supports this for structured agents - we just need to extend it for web search agents.

