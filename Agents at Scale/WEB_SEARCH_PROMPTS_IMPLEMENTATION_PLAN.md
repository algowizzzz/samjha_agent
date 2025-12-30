# Web Search Prompts Implementation Plan

## Overview

Create 6 web search prompts (4 new + 2 clones from structured) and integrate with admin panel.

**No shared prompts** - each agent type has its own independent prompts.

---

## Prompt List

### Structured Agents (Existing - 6 prompts)
1. `decider.md`
2. `ask_user_clarification.md`
3. `nl_to_sql_planner.md`
4. `query_result_evaluator.md`
5. `response_commentary.md`
6. `sql_plan_updater.md`

### Web Search Agents (New - 6 prompts)
1. `web_research_decider.md` - NEW (Gate prompt)
2. `web_research_ask_user_clarification.md` - CLONE (from structured)
3. `web_research_synthesis.md` - NEW (Final answer generation)
4. `web_research_claim_extraction.md` - NEW (Extract claims from sources)
5. `web_research_conflict_detection.md` - NEW (Detect conflicts)
6. `web_research_response_commentary.md` - CLONE (from structured)

**Total: 12 prompts** (6 structured + 6 web search)

---

## Implementation Steps

### Phase 1: Create Prompt Files

#### Step 1.1: Create New Prompts (4 files)

**1.1.1: web_research_decider.md**
- Location: `external/config/prompts/web_research_decider.md`
- Source: Based on `decider.md` but adapted for ResearchSpec
- Size: ~500-800 lines
- Key changes:
  - Replace QuerySpec → ResearchSpec
  - Replace SQL tools → Tavily tools
  - Replace investigation_plan → research_plan
  - Add evidence_pack handling
  - Add conflict/gap detection logic

**1.1.2: web_research_synthesis.md**
- Location: `external/config/prompts/web_research_synthesis.md`
- Source: New (combines query_result_evaluator + response_commentary concepts)
- Size: ~200-300 lines
- Purpose: Generate final answer from EvidencePack
- Sections:
  - Role: Synthesize evidence into answer
  - Input: evidence_pack, research_spec, user_query
  - Output: Report with citations
  - Conflict handling
  - Confidence levels

**1.1.3: web_research_claim_extraction.md**
- Location: `external/config/prompts/web_research_claim_extraction.md`
- Source: New
- Size: ~150-200 lines
- Purpose: Extract claims from source content (Executor node)
- Sections:
  - Role: Extract factual claims
  - Input: Source content, research_spec scope
  - Output: List of claims with confidence

**1.1.4: web_research_conflict_detection.md**
- Location: `external/config/prompts/web_research_conflict_detection.md`
- Source: New
- Size: ~150-200 lines
- Purpose: Detect conflicts between claims (Executor node)
- Sections:
  - Role: Detect contradictory claims
  - Input: List of claims
  - Output: Conflicts list with severity

#### Step 1.2: Clone Shared Prompts (2 files)

**1.2.1: web_research_ask_user_clarification.md**
- Location: `external/config/prompts/web_research_ask_user_clarification.md`
- Source: Clone from `ask_user_clarification.md`
- Changes:
  - Update examples to use ResearchSpec instead of QuerySpec
  - Update terminology (research_spec vs query_spec)
  - Keep same structure/format

**1.2.2: web_research_response_commentary.md**
- Location: `external/config/prompts/web_research_response_commentary.md`
- Source: Clone from `response_commentary.md`
- Changes:
  - Update examples to use EvidencePack instead of SQL results
  - Update terminology (sources vs rows)
  - Keep same structure/format

---

### Phase 2: Database Setup

#### Step 2.1: Import Prompts to Database

**Action:** Run import script or manually insert via admin panel

**Prompts to import:**
1. `web_research_decider` (category: `web_search`)
2. `web_research_ask_user_clarification` (category: `web_search`)
3. `web_research_synthesis` (category: `web_search`)
4. `web_research_claim_extraction` (category: `web_search`)
5. `web_research_conflict_detection` (category: `web_search`)
6. `web_research_response_commentary` (category: `web_search`)

**Method:** Use existing `import_prompts_from_files()` function or add via admin panel

---

### Phase 3: Admin Panel Updates

#### Step 3.1: Update Admin Panel HTML

**File:** `web/templates/admin.html`

**Changes:**
1. Update "Web-based Prompts" section (currently placeholder)
2. Add JavaScript to load `web_search` category prompts
3. Update `loadPrompts()` function to handle `web_search` category

**Code changes:**

```javascript
// In admin.html, find "Web-based Prompts" section
// Replace placeholder with:
<div id="web-search-prompts-content" class="agent-section-content" style="display: none;">
    <div class="loading" id="web-search-prompts-loading">Loading prompts...</div>
    <ul class="prompt-list" id="web-search-prompts-list" style="display: none;">
        <!-- Prompts will be loaded here -->
    </ul>
</div>

// Update loadPrompts() function to handle 'web_search' category
async function loadPrompts(category) {
    // ... existing code ...
    if (category === 'web_search') {
        const loadingEl = document.getElementById('web-search-prompts-loading');
        const listEl = document.getElementById('web-search-prompts-list');
        // ... load logic ...
    }
}

// Update toggleSection() to load web_search prompts
function toggleSection(sectionId) {
    // ... existing code ...
    if (sectionId === 'system-prompts' && !content.dataset.loaded) {
        loadPrompts('structured');
        loadPrompts('web_search'); // ADD THIS
        content.dataset.loaded = 'true';
    }
}
```

#### Step 3.2: Update Prompt Metadata

**File:** `external/agent/persistence.py`

**Changes:** Update `PROMPT_METADATA` to include web search prompts

```python
PROMPT_METADATA = {
    # ... existing structured prompts ...
    
    # Web Search Prompts
    "web_research_decider": {
        "display_name": "Web Research Decider",
        "description": "Gate prompt that creates ResearchSpec and decides ASK_USER/EXECUTE/BLOCK for web research queries."
    },
    "web_research_ask_user_clarification": {
        "display_name": "Web Research Clarification",
        "description": "Generates clarification questions when web research agent needs more information."
    },
    "web_research_synthesis": {
        "display_name": "Web Research Synthesis",
        "description": "Generates final answer from EvidencePack with citations and conflict handling."
    },
    "web_research_claim_extraction": {
        "display_name": "Claim Extraction",
        "description": "Extracts factual claims from web source content."
    },
    "web_research_conflict_detection": {
        "display_name": "Conflict Detection",
        "description": "Detects conflicts between claims from different sources."
    },
    "web_research_response_commentary": {
        "display_name": "Web Research Commentary",
        "description": "Generates natural language explanations of web research results."
    }
}
```

---

### Phase 4: Backend Integration

#### Step 4.1: Update Prompt Loading Logic

**File:** `external/agent/web_research_decider.py` (to be created)

```python
def load_web_research_decider_prompt() -> str:
    """Load web research decider prompt from DB or file."""
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
    logger.warning("Web research decider prompt not found, using fallback")
    return "# WEB RESEARCH DECIDER PROMPT\n\nOutput JSON only."
```

**Repeat for:**
- `web_research_synthesis.md` → `web_research_synthesis.py`
- `web_research_claim_extraction.md` → `web_research_executor_nodes.py`
- `web_research_conflict_detection.md` → `web_research_executor_nodes.py`
- `web_research_ask_user_clarification.md` → `web_research_decider.py`
- `web_research_response_commentary.md` → `web_research_synthesis.py`

#### Step 4.2: Update Import Function

**File:** `external/agent/persistence.py`

**Function:** `import_prompts_from_files()`

**Changes:** Ensure it handles `web_search` category

```python
def import_prompts_from_files(db) -> int:
    """Import prompts from external/config/prompts/*.md into DB."""
    from pathlib import Path
    prompts_dir = Path("external/config/prompts")
    if not prompts_dir.exists():
        return 0
    
    imported = 0
    for prompt_file in prompts_dir.glob("*.md"):
        name = prompt_file.stem
        content = prompt_file.read_text(encoding="utf-8", errors="replace")
        
        # Determine category
        if name.startswith("web_research_"):
            category = "web_search"
        elif name in ["decider", "nl_to_sql_planner", "query_result_evaluator", 
                      "sql_plan_updater", "ask_user_clarification", "response_commentary"]:
            category = "structured"
        else:
            category = "structured"  # Default
        
        p = db.get(Prompt, name)
        if p is None:
            p = Prompt(name=name, category=category, current_content=content)
            db.add(p)
            imported += 1
            logger.info(f"Imported prompt: {name} (category: {category})")
    
    return imported
```

---

### Phase 5: Testing

#### Step 5.1: Admin Panel Testing

**Test Cases:**
1. ✅ Load web search prompts in admin panel
2. ✅ Edit web search prompt
3. ✅ Save changes
4. ✅ Verify changes persist
5. ✅ Verify structured prompts unchanged
6. ✅ Verify category filtering works

#### Step 5.2: Runtime Testing

**Test Cases:**
1. ✅ Load prompt from DB (after admin edit)
2. ✅ Fallback to file if not in DB
3. ✅ Verify correct prompt used for web search agent
4. ✅ Verify structured agent still uses structured prompts

---

## File Structure

### New Files to Create

```
external/config/prompts/
├── web_research_decider.md                    [NEW]
├── web_research_ask_user_clarification.md     [CLONE]
├── web_research_synthesis.md                  [NEW]
├── web_research_claim_extraction.md           [NEW]
├── web_research_conflict_detection.md         [NEW]
└── web_research_response_commentary.md        [CLONE]
```

### Files to Modify

```
web/templates/admin.html                        [UPDATE - add web_search section]
external/agent/persistence.py                  [UPDATE - add metadata, import logic]
external/agent/web_research_decider.py          [UPDATE - prompt loading]
external/agent/web_research_executor_nodes.py   [UPDATE - prompt loading]
external/agent/web_research_synthesis.py        [UPDATE - prompt loading]
```

---

## Implementation Order

### Day 1: Prompt Creation
1. ✅ Create `web_research_decider.md` (based on decider.md)
2. ✅ Create `web_research_synthesis.md` (new)
3. ✅ Create `web_research_claim_extraction.md` (new)
4. ✅ Create `web_research_conflict_detection.md` (new)
5. ✅ Clone `ask_user_clarification.md` → `web_research_ask_user_clarification.md`
6. ✅ Clone `response_commentary.md` → `web_research_response_commentary.md`

### Day 2: Database & Admin Panel
1. ✅ Import prompts to database (via import function or admin panel)
2. ✅ Update `persistence.py` metadata
3. ✅ Update admin panel HTML (web_search section)
4. ✅ Update JavaScript (loadPrompts function)
5. ✅ Test admin panel editing

### Day 3: Backend Integration
1. ✅ Update prompt loading functions
2. ✅ Update import function
3. ✅ Test runtime loading
4. ✅ Verify fallback logic

### Day 4: Testing & Validation
1. ✅ Test all prompts load correctly
2. ✅ Test admin panel editing
3. ✅ Test runtime usage
4. ✅ Verify no regressions in structured agents

---

## Success Criteria

### ✅ Prompts Created
- [ ] 6 web search prompts exist in `external/config/prompts/`
- [ ] All prompts have proper structure and content
- [ ] Clones are independent (no shared references)

### ✅ Database Integration
- [ ] All 6 prompts imported to database
- [ ] Category set to `web_search`
- [ ] Metadata added to PROMPT_METADATA

### ✅ Admin Panel
- [ ] "Web Search Prompts" section displays
- [ ] All 6 prompts listed
- [ ] Editing works (save/load)
- [ ] Category filtering works

### ✅ Runtime Integration
- [ ] Prompts load from DB (after admin edit)
- [ ] Fallback to file works (if not in DB)
- [ ] Correct prompts used for web search agents
- [ ] Structured agents unaffected

---

## Notes

### No Prompt Creation UI
- User requirement: Only editing, no creation
- Prompts created via file system → imported to DB
- Admin panel only edits existing prompts

### Independent Prompts
- No shared prompts between agent types
- Each agent type has its own set
- Changes to one don't affect the other

### Category System
- `structured` - Structured agent prompts
- `web_search` - Web search agent prompts
- Categories used for filtering in admin panel

---

## Next Steps After Implementation

1. **Create web search agent handler** (uses these prompts)
2. **Test end-to-end flow** (Decider → Executor → Synthesis)
3. **Tune prompts** based on real usage
4. **Document prompt editing** for admins

---

## Estimated Time

- **Prompt Creation:** 4-6 hours (writing/adapting content)
- **Database Setup:** 1 hour
- **Admin Panel Updates:** 2-3 hours
- **Backend Integration:** 2-3 hours
- **Testing:** 2-3 hours

**Total: 11-16 hours** (1.5-2 days)

