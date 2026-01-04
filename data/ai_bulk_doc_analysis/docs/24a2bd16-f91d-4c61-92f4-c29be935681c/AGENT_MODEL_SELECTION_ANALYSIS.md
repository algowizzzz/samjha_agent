# Analysis: Adding Model Selection Dropdown to Agent Management

## Overview

Add a comprehensive model selection dropdown (similar to workflow template editor) to the "Manage Agents - Structured" section where users create/edit agent instances.

## Current State

### Workflow Template Editor (Reference Implementation)

**Location**: `web/templates/admin.html` (lines 1693-1709)

**Current Implementation**:
- Full dropdown with 10 model options
- Models include: Haiku 4.5, Claude 3 Haiku, Claude 3.5 Haiku, Sonnet 4.5, Sonnet 4, Claude 3.7 Sonnet, Opus 4.5, Opus 4.1, Opus 4, Claude 3 Opus
- Each option has descriptive labels (e.g., "Fastest model for daily tasks")
- Styled with `workflow-step-field-row` class
- Includes helpful small text below dropdown

**HTML Structure**:
```html
<div class="workflow-step-field-row">
    <div class="form-group" style="margin-bottom: 0;">
        <label>Model</label>
        <select class="workflow-step-model">
            <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5 (Fastest model for daily tasks)</option>
            <option value="claude-3-haiku-20240307">Claude 3 Haiku (Fast, cost-effective)</option>
            <!-- ... 8 more options ... -->
        </select>
        <small>Haiku (default) for fast, cost-effective tasks. Sonnet for balanced performance. Opus for complex analysis.</small>
    </div>
</div>
```

### Agent Management - Structured (Current Implementation)

**Location**: `web/templates/admin.html` (lines 642-651)

**Current Implementation**:
- Limited dropdown with only 4 model options
- Models: Sonnet 4 (default), Claude 3.5 Sonnet, Claude 3 Sonnet, Claude 3 Haiku
- Basic styling with standard `form-group` class
- No descriptive labels or helpful text

**HTML Structure**:
```html
<div class="form-group">
    <label>LLM Model *</label>
    <select name="model" required>
        <option value="claude-sonnet-4-20250514" selected>Claude Sonnet 4 (Recommended)</option>
        <option value="claude-3-5-sonnet-20240620">Claude 3.5 Sonnet</option>
        <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
        <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
    </select>
    <small>Default: Sonnet 4 enables advanced reasoning capabilities</small>
</div>
```

## Required Changes

### 1. Frontend Changes (HTML Template)

**File**: `web/templates/admin.html`

#### 1.1 Update Structured Agent Form (Lines 642-651)

**Replace** the current limited dropdown with the full model selection dropdown matching workflow editor:

**Current**:
```html
<div class="form-group">
    <label>LLM Model *</label>
    <select name="model" required>
        <option value="claude-sonnet-4-20250514" selected>Claude Sonnet 4 (Recommended)</option>
        <option value="claude-3-5-sonnet-20240620">Claude 3.5 Sonnet</option>
        <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
        <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
    </select>
    <small>Default: Sonnet 4 enables advanced reasoning capabilities</small>
</div>
```

**Replace With**:
```html
<div class="form-group">
    <label>LLM Model *</label>
    <select name="model" required>
        <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5 (Fastest model for daily tasks)</option>
        <option value="claude-3-haiku-20240307">Claude 3 Haiku (Fast, cost-effective)</option>
        <option value="claude-3-5-haiku-20241022">Claude 3.5 Haiku</option>
        <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5 (Smart, efficient model for everyday use)</option>
        <option value="claude-sonnet-4-20250514" selected>Claude Sonnet 4 (Recommended)</option>
        <option value="claude-3-7-sonnet-20250219">Claude 3.7 Sonnet</option>
        <option value="claude-opus-4-5-20251101">Claude Opus 4.5 (Powerful, large model for complex challenges)</option>
        <option value="claude-opus-4-1-20250805">Claude Opus 4.1</option>
        <option value="claude-opus-4-20250514">Claude Opus 4</option>
        <option value="claude-3-opus-20240229">Claude 3 Opus</option>
    </select>
    <small style="display: block; margin-top: 4px; color: #6c757d; font-size: 12px;">Haiku for fast, cost-effective tasks. Sonnet for balanced performance. Opus for complex analysis.</small>
</div>
```

**Key Changes**:
- Add 6 more model options (total 10 options)
- Add descriptive labels to each option
- Update help text to match workflow editor style
- Keep `selected` on Sonnet 4 as default (recommended)

#### 1.2 Update External Agent Form (Lines 736-744)

**Same replacement needed** for external agents form to maintain consistency.

**Current**:
```html
<div class="form-group">
    <label>LLM Model *</label>
    <select name="model" required>
        <option value="claude-sonnet-4-20250514" selected>Claude Sonnet 4 (Recommended)</option>
        <option value="claude-3-5-sonnet-20240620">Claude 3.5 Sonnet</option>
        <option value="claude-3-sonnet-20240229">Claude 3 Sonnet</option>
        <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
    </select>
</div>
```

**Replace With**: Same full dropdown as structured agents (10 options with descriptive labels).

### 2. JavaScript Changes

**File**: `web/templates/admin.html`

#### 2.1 Update `editAgent()` Function (Lines 1306-1309)

**Current Code**:
```javascript
const modelSelect = form.querySelector('[name="model"]');
if (modelSelect) {
    modelSelect.value = agent.model || 'claude-sonnet-4-20250514';
}
```

**Status**: ✅ **No changes needed** - This code already handles any model value correctly. It will work with the new model options.

**Note**: The function sets the dropdown value from `agent.model` in the database. As long as the model value matches one of the option values, it will work correctly.

#### 2.2 Update `saveAgent()` Function (Lines 1354-1390)

**Current Code**:
```javascript
const model = formData.get('model');
// ... later ...
finalFormData.append('model', model);
```

**Status**: ✅ **No changes needed** - The function already extracts and sends the model value to the backend. It will work with any model value from the dropdown.

### 3. Backend Changes

#### 3.1 API Endpoint - Create Agent (POST `/api/admin/agents`)

**File**: `routes/admin_routes.py` (lines 348-557)

**Current Implementation**:
- Extracts `model` from form: `model = (request.form.get("model") or "").strip()`
- Passes to `create_agent_db()` with default fallback

**Status**: ✅ **No changes needed** - The endpoint already accepts any model value and stores it in the database.

**Code Reference** (line ~400):
```python
model = (request.form.get("model") or "").strip()
# ... later ...
agent = create_agent_db(
    db,
    agent_id=agent_id,
    name=name,
    agent_type=agent_type,
    description=description,
    domain_file=domain_filename,
    domain_content=domain_text,
    data_folder=data_folder,
    model=model,  # Already handles any model value
)
```

#### 3.2 API Endpoint - Update Agent (PUT `/api/admin/agents/<agent_id>`)

**File**: `routes/admin_routes.py` (lines 139-283)

**Current Implementation**:
- Extracts `model` from form: `model = (request.form.get("model") or "").strip()`
- Updates via `update_agent_db()` if model is provided

**Status**: ✅ **No changes needed** - The endpoint already accepts any model value and updates it in the database.

**Code Reference** (lines 229-244):
```python
# Get model from form (optional update)
model = (request.form.get("model") or "").strip()
# ... later ...
if model:
    update_data["model"] = model
```

#### 3.3 Database Persistence Layer

**File**: `external/agent/persistence.py`

**Functions**:
- `create_agent_db()` (lines 390-427): ✅ Already accepts `model` parameter
- `update_agent_db()` (lines 430-458): ✅ Already accepts `model` parameter

**Status**: ✅ **No changes needed** - Database layer already supports any model value.

**Code Reference**:
```python
def create_agent_db(
    ...
    model: Optional[str] = None,
) -> Agent:
    # Default to Sonnet if not specified
    if model is None:
        model = "claude-3-sonnet-20240229"
    # ... stores model in Agent.model field
```

### 4. Database Schema

**File**: `core/db/models.py` (Agent model)

**Status**: ✅ **No changes needed** - The `Agent` model already has a `model` field that stores the model identifier as a string. It can store any model value.

**Expected Schema**:
```python
class Agent(Base):
    ...
    model: Optional[str] = Column(String, nullable=True)
    ...
```

### 5. Agent Loading/Usage

**File**: `external/agent/parquet_agent.py` (lines 200-228)

**Current Implementation**:
- Loads agent from database
- Extracts `model` field: `agent_model = agent.get("model") or "claude-sonnet-4-20250514"`
- Passes to state as `agent_model`

**Status**: ✅ **No changes needed** - The code already loads any model value from the database and uses it. The decider and executor will use whatever model is stored.

**Code Reference**:
```python
agent_model = agent.get("model") or "claude-sonnet-4-20250514"  # Default fallback
# ... later ...
"agent_model": agent_model  # Stored in state
```

### 6. Decider Model Support

**File**: `external/agent/decider.py` (lines 158-180)

**Current Implementation**:
- Reads `agent_model` from state
- Determines model capabilities (max_tokens, thinking support) based on model name
- Uses model-specific settings

**Status**: ⚠️ **Review needed** - Check if all new model options are properly handled.

**Current Model Detection** (lines 170-180):
```python
if model:
    model_lower = model.lower()
    if "sonnet" in model_lower:
        model_max_tokens = 8192
        model_supports_thinking = True
    elif "haiku" in model_lower:
        model_max_tokens = 4096
        model_supports_thinking = False
    elif "opus" in model_lower:
        model_max_tokens = 4096
        model_supports_thinking = True
```

**Analysis**:
- ✅ Haiku models: Correctly detected (all Haiku models contain "haiku")
- ✅ Sonnet models: Correctly detected (all Sonnet models contain "sonnet")
- ✅ Opus models: Correctly detected (all Opus models contain "opus")
- ✅ Max tokens: Uses conservative defaults (4096 for Haiku/Opus, 8192 for Sonnet)
- ✅ Thinking support: Correctly disabled for Haiku, enabled for Sonnet/Opus

**Recommendation**: ✅ **No changes needed** - The pattern matching works for all new model options.

## Summary of Changes Required

### ✅ Changes Needed

1. **Frontend HTML** (`web/templates/admin.html`):
   - Replace limited dropdown (4 options) with full dropdown (10 options) in structured agent form (lines 642-651)
   - Replace limited dropdown (4 options) with full dropdown (10 options) in external agent form (lines 736-744)
   - Add descriptive labels to each option
   - Update help text to match workflow editor style

### ✅ No Changes Needed (Already Works)

1. **JavaScript** (`web/templates/admin.html`):
   - `editAgent()` function - already handles any model value
   - `saveAgent()` function - already extracts and sends model value

2. **Backend API** (`routes/admin_routes.py`):
   - POST `/api/admin/agents` - already accepts and stores any model value
   - PUT `/api/admin/agents/<agent_id>` - already accepts and updates any model value

3. **Database Layer** (`external/agent/persistence.py`):
   - `create_agent_db()` - already accepts model parameter
   - `update_agent_db()` - already accepts model parameter

4. **Agent Loading** (`external/agent/parquet_agent.py`):
   - Already loads model from database and uses it

5. **Decider Model Support** (`external/agent/decider.py`):
   - Pattern matching works for all new model options
   - Max tokens and thinking support correctly determined

## Implementation Steps

### Step 1: Update HTML Template
1. Open `web/templates/admin.html`
2. Find structured agent form (line ~642)
3. Replace model dropdown with full 10-option dropdown
4. Find external agent form (line ~736)
5. Replace model dropdown with full 10-option dropdown

### Step 2: Test
1. Create new structured agent with different model options
2. Verify model is saved correctly in database
3. Edit existing agent and change model
4. Verify model update works
5. Test agent execution with different models to ensure they work

### Step 3: Verify Model Support
1. Test each model option to ensure decider correctly detects capabilities
2. Verify max_tokens and thinking support are set correctly for each model
3. Test agent execution with Haiku, Sonnet, and Opus models

## Model Options Reference

### Full List of Models (10 options)

1. `claude-haiku-4-5-20251001` - Claude Haiku 4.5 (Fastest model for daily tasks)
2. `claude-3-haiku-20240307` - Claude 3 Haiku (Fast, cost-effective)
3. `claude-3-5-haiku-20241022` - Claude 3.5 Haiku
4. `claude-sonnet-4-5-20250929` - Claude Sonnet 4.5 (Smart, efficient model for everyday use)
5. `claude-sonnet-4-20250514` - Claude Sonnet 4 (Recommended) ⭐ Default
6. `claude-3-7-sonnet-20250219` - Claude 3.7 Sonnet
7. `claude-opus-4-5-20251101` - Claude Opus 4.5 (Powerful, large model for complex challenges)
8. `claude-opus-4-1-20250805` - Claude Opus 4.1
9. `claude-opus-4-20250514` - Claude Opus 4
10. `claude-3-opus-20240229` - Claude 3 Opus

## Notes

1. **Default Selection**: Keep Sonnet 4 (`claude-sonnet-4-20250514`) as the default/selected option since it's recommended and supports thinking.

2. **Backward Compatibility**: Existing agents with old model values (e.g., `claude-3-haiku-20240307`) will continue to work. The dropdown will show the correct option when editing.

3. **Model Detection**: The decider uses pattern matching (`"haiku" in model_lower`, etc.), so all new model options will be correctly categorized.

4. **Consistency**: This change makes agent management consistent with workflow template editor, providing users with the same model selection experience across both features.

