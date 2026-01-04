# Vision Ingestion Implementation Plan

## Overview
This plan covers all changes needed to:
1. Remove unsupported "Images" checkbox
2. Add configurable Vision Ingestion with Advanced Options
3. Implement per-page processing (one LLM call per page)
4. Make all vision parameters configurable

---

## BACKEND CHANGES

### 1. Database Model Updates
**File:** `external/ai_bulk_doc_analysis/models.py`

**Status:** ✅ No changes needed - `metadata_json` field already exists

**Note:** We'll store vision config in `metadata_json.vision_config` to avoid migrations.

---

### 2. Ingestion Service Updates
**File:** `external/ai_bulk_doc_analysis/ingestion_service.py`

#### 2.1 Update `create_ingestion_profile()` method
**Location:** Lines 54-112

**Changes:**
- Add parameter: `vision_config: Optional[Dict] = None`
- Store vision_config in `metadata_json` when creating profile
- Default vision_config structure:
  ```python
  {
      "dpi": 200,
      "image_format": "PNG",
      "jpeg_quality": 85,
      "model": "claude-3-opus-20240229",
      "max_tokens": 4096,
      "temperature": 0.0
  }
  ```

**Code changes:**
```python
def create_ingestion_profile(
    self,
    name: str,
    accepted_input_types: List[str],
    mode: str,
    vision_prompt: Optional[str] = None,
    vision_config: Optional[Dict] = None  # NEW
) -> IngestionProfile:
    # ... existing validation ...
    
    # Set default vision_config if mode is vision
    if mode == 'vision' and vision_config is None:
        vision_config = {
            "dpi": 200,
            "image_format": "PNG",
            "jpeg_quality": 85,
            "model": "claude-3-opus-20240229",
            "max_tokens": 4096,
            "temperature": 0.0
        }
    
    # Create profile with metadata
    metadata = {}
    if vision_config:
        metadata["vision_config"] = vision_config
    
    profile = IngestionProfile(
        # ... existing fields ...
        metadata_json=metadata
    )
```

#### 2.2 Rewrite `_ingest_vision()` method
**Location:** Lines 319-405

**Major Changes:**
1. Change signature to accept `IngestionProfile` instead of just `vision_prompt`
2. Extract vision_config from `metadata_json`
3. Process one page at a time with separate LLM calls
4. Add page markers in output
5. Track tokens per page
6. Handle errors per page (continue on error)

**New method signature:**
```python
def _ingest_vision(
    self, 
    file_path: Path, 
    ingestion_profile: IngestionProfile  # Changed from vision_prompt: str
) -> Tuple[str, Dict]:
```

**Key logic changes:**
```python
# Extract config
vision_config = ingestion_profile.metadata_json.get("vision_config", {})
dpi = vision_config.get("dpi", 200)
image_format = vision_config.get("image_format", "PNG").upper()
jpeg_quality = vision_config.get("jpeg_quality", 85)
model = vision_config.get("model", "claude-3-opus-20240229")
max_tokens = vision_config.get("max_tokens", 4096)
temperature = vision_config.get("temperature", 0.0)
vision_prompt = ingestion_profile.vision_prompt

# Convert PDF with configurable DPI
images = convert_from_path(str(file_path), dpi=dpi, fmt=image_format.lower())

# Process one page at a time
r0_parts = []
r0_parts.append(f"# Document: {file_path.name}\n\n")
total_input_tokens = 0
total_output_tokens = 0

for page_num, img in enumerate(images, start=1):
    # Encode single image with configurable format
    # Make API call for this page only
    # Append result with page marker
    # Track tokens
    # Handle errors per page

r0_content = ''.join(r0_parts)
```

#### 2.3 Update `ingest_file()` method
**Location:** Line 157

**Change:**
```python
# FROM:
return self._ingest_vision(file_path, ingestion_profile.vision_prompt)

# TO:
return self._ingest_vision(file_path, ingestion_profile)
```

#### 2.4 Update `estimate_ingestion_tokens()` method
**Location:** Lines 407-447

**Changes:**
- Extract vision_config from metadata_json
- Account for configurable DPI in token estimation
- Multiply by page count (since we process per-page)

```python
elif ingestion_profile.mode == 'vision':
    vision_config = ingestion_profile.metadata_json.get("vision_config", {})
    dpi = vision_config.get("dpi", 200)
    
    # Higher DPI = more tokens per image
    tokens_per_image = int(1000 * (dpi / 200))
    
    # Get page count
    images = convert_from_path(str(file_path), first_page=1, last_page=1, dpi=dpi)
    image_count = len(images) if images else 1
    
    # Estimate: prompt tokens + (image tokens * page count)
    prompt_tokens = len(ingestion_profile.vision_prompt or "") // 4
    image_tokens = image_count * tokens_per_image * image_count  # Per page
    return prompt_tokens + image_tokens
```

---

### 3. API Endpoint Updates
**File:** `external/ai_bulk_doc_analysis/blueprint.py`

#### 3.1 Create Ingestion Profile Endpoint
**Location:** Lines 777-824

**Changes:**
- Accept `vision_config` in request body
- Pass `vision_config` to `create_ingestion_profile()`
- Return `vision_config` in response

**Request body addition:**
```python
{
    "name": "...",
    "accepted_input_types": [...],
    "mode": "vision",
    "vision_prompt": "...",
    "vision_config": {  # NEW
        "dpi": 200,
        "image_format": "PNG",
        "jpeg_quality": 85,
        "model": "claude-3-opus-20240229",
        "max_tokens": 4096,
        "temperature": 0.0
    }
}
```

**Code changes:**
```python
vision_config = data.get("vision_config")  # NEW

profile = ingestion_service.create_ingestion_profile(
    name=name,
    accepted_input_types=accepted_input_types,
    mode=mode,
    vision_prompt=vision_prompt,
    vision_config=vision_config  # NEW
)
```

#### 3.2 Get Ingestion Profile Endpoint
**Location:** Lines 826-851

**Changes:**
- Return `vision_config` from `metadata_json` in response

**Response addition:**
```python
return jsonify({
    "ingestion_profile_id": profile.ingestion_profile_id,
    "name": profile.name,
    "accepted_input_types": profile.accepted_input_types,
    "mode": profile.mode,
    "vision_prompt": profile.vision_prompt,
    "vision_config": profile.metadata_json.get("vision_config", {}),  # NEW
    "created_at": profile.created_at.isoformat() if profile.created_at else None,
})
```

#### 3.3 Update Ingestion Profile Endpoint
**Location:** Lines 853-902

**Changes:**
- Accept `vision_config` in request body
- Update `metadata_json` with new vision_config
- Merge with existing config if partial update

**Code changes:**
```python
vision_config = data.get("vision_config")  # NEW

# Update metadata_json
if vision_config is not None:
    if not isinstance(db_profile.metadata_json, dict):
        db_profile.metadata_json = {}
    db_profile.metadata_json["vision_config"] = vision_config
```

---

### 4. Default Vision Prompt Constant
**File:** `external/ai_bulk_doc_analysis/ingestion_service.py` (top of file)

**Add constant:**
```python
DEFAULT_VISION_PROMPT = """Extract all text from these document pages. Preserve the structure, formatting, and layout as much as possible. Include all tables, lists, and paragraphs. Convert the content to markdown format with appropriate headings, tables, and formatting."""
```

**Usage:** Pre-fill in UI and use as fallback if prompt not provided.

---

## FRONTEND CHANGES

### 1. Remove Images Checkbox
**File:** `web/templates/admin.html`

**Location:** Lines 927-930

**Action:** Delete the entire "Images" checkbox label block

**Code to remove:**
```html
<label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
    <input type="checkbox" name="doc_types" value="Images">
    <span>Images</span>
</label>
```

**Also update help text** (line 932):
```html
<!-- FROM: -->
<small>Select the file formats this agentic workflow can process. PDF is recommended for documents with complex formatting, while DOCX and TXT work well for simpler documents. CSV and Images may require specialized processing.</small>

<!-- TO: -->
<small>Select the file formats this agentic workflow can process. PDF is recommended for documents with complex formatting, while DOCX and TXT work well for simpler documents. CSV may require specialized processing.</small>
```

---

### 2. Add Advanced Options Section
**File:** `web/templates/admin.html`

**Location:** After line 933 (after "Accepted Document Types" section, before "Processing Steps")

**New HTML structure:**
```html
<!-- Advanced Options -->
<div class="form-group" id="advanced-options-section">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <label>Advanced Options</label>
    </div>
    
    <!-- Vision Ingestion Toggle -->
    <div style="margin-bottom: 16px;">
        <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
            <input type="checkbox" id="vision-ingestion-enabled" name="vision_enabled">
            <span><strong>Enable Vision Ingestion</strong></span>
        </label>
        <small style="display: block; margin-top: 4px; margin-left: 24px; color: #6c757d;">
            Use AI vision model to transcribe documents by converting PDF pages to images. Useful for scanned documents or complex layouts.
        </small>
    </div>
    
    <!-- Vision Configuration (shown when enabled) -->
    <div id="vision-config-container" style="display: none; margin-left: 24px; padding: 16px; background: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6;">
        
        <!-- Vision Prompt -->
        <div class="form-group" style="margin-bottom: 16px;">
            <label>Vision Prompt *</label>
            <textarea id="vision-prompt" name="vision_prompt" rows="4" style="width: 100%; font-family: monospace; font-size: 13px;" placeholder="Enter instructions for the vision model..."></textarea>
            <small style="display: block; margin-top: 4px; color: #6c757d;">
                Instructions for extracting text from document images. Default prompt will be used if empty.
            </small>
        </div>
        
        <!-- Vision Configuration Grid -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
            
            <!-- DPI -->
            <div class="form-group">
                <label>DPI (Resolution)</label>
                <select id="vision-dpi" name="vision_dpi" style="width: 100%;">
                    <option value="150">150 DPI (Faster, smaller files)</option>
                    <option value="200" selected>200 DPI (Recommended - balanced)</option>
                    <option value="300">300 DPI (Higher quality, larger files)</option>
                </select>
                <small style="display: block; margin-top: 4px; color: #6c757d;">
                    Higher DPI = better quality but more tokens/cost
                </small>
            </div>
            
            <!-- Image Format -->
            <div class="form-group">
                <label>Image Format</label>
                <select id="vision-image-format" name="vision_image_format" style="width: 100%;">
                    <option value="PNG" selected>PNG (Lossless, recommended)</option>
                    <option value="JPEG">JPEG (Smaller files, lossy)</option>
                </select>
                <small style="display: block; margin-top: 4px; color: #6c757d;">
                    PNG preserves quality, JPEG reduces file size
                </small>
            </div>
            
            <!-- JPEG Quality (shown only if JPEG selected) -->
            <div class="form-group" id="jpeg-quality-container" style="display: none;">
                <label>JPEG Quality</label>
                <input type="range" id="vision-jpeg-quality" name="vision_jpeg_quality" min="50" max="100" value="85" style="width: 100%;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6c757d; margin-top: 4px;">
                    <span>50 (Smaller)</span>
                    <span id="jpeg-quality-value">85</span>
                    <span>100 (Better)</span>
                </div>
            </div>
            
            <!-- Model Selection -->
            <div class="form-group">
                <label>Vision Model</label>
                <select id="vision-model" name="vision_model" style="width: 100%;">
                    <option value="claude-3-opus-20240229" selected>Claude 3 Opus (Best quality)</option>
                    <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet (Balanced)</option>
                    <option value="claude-3-5-haiku-20241022">Claude 3.5 Haiku (Fastest)</option>
                </select>
                <small style="display: block; margin-top: 4px; color: #6c757d;">
                    Opus = best quality, Haiku = fastest/cheapest
                </small>
            </div>
            
            <!-- Max Tokens -->
            <div class="form-group">
                <label>Max Tokens (per page)</label>
                <input type="number" id="vision-max-tokens" name="vision_max_tokens" value="4096" min="1024" max="16384" step="1024" style="width: 100%;">
                <small style="display: block; margin-top: 4px; color: #6c757d;">
                    Maximum tokens per page response (1024-16384)
                </small>
            </div>
            
            <!-- Temperature -->
            <div class="form-group">
                <label>Temperature</label>
                <input type="range" id="vision-temperature" name="vision_temperature" min="0" max="100" value="0" step="10" style="width: 100%;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6c757d; margin-top: 4px;">
                    <span>0.0 (Deterministic)</span>
                    <span id="temperature-value">0.0</span>
                    <span>1.0 (Creative)</span>
                </div>
                <small style="display: block; margin-top: 4px; color: #6c757d;">
                    Lower = more consistent extraction (recommended: 0.0)
                </small>
            </div>
            
        </div>
    </div>
</div>
```

---

### 3. JavaScript Functions
**File:** `web/templates/admin.html`

#### 3.1 Toggle Vision Config Visibility
**Location:** Add after existing JavaScript functions

```javascript
// Toggle vision config visibility
document.getElementById('vision-ingestion-enabled')?.addEventListener('change', function(e) {
    const container = document.getElementById('vision-config-container');
    if (container) {
        container.style.display = e.target.checked ? 'block' : 'none';
        // Set required on prompt if enabled
        const promptField = document.getElementById('vision-prompt');
        if (promptField) {
            promptField.required = e.target.checked;
        }
    }
});

// Update JPEG quality display
document.getElementById('vision-jpeg-quality')?.addEventListener('input', function(e) {
    const display = document.getElementById('jpeg-quality-value');
    if (display) {
        display.textContent = e.target.value;
    }
});

// Show/hide JPEG quality based on format
document.getElementById('vision-image-format')?.addEventListener('change', function(e) {
    const jpegContainer = document.getElementById('jpeg-quality-container');
    if (jpegContainer) {
        jpegContainer.style.display = e.target.value === 'JPEG' ? 'block' : 'none';
    }
});

// Update temperature display
document.getElementById('vision-temperature')?.addEventListener('input', function(e) {
    const display = document.getElementById('temperature-value');
    if (display) {
        display.textContent = (parseInt(e.target.value) / 100).toFixed(1);
    }
});
```

#### 3.2 Default Vision Prompt
**Location:** Add constant at top of script section

```javascript
const DEFAULT_VISION_PROMPT = `Extract all text from these document pages. Preserve the structure, formatting, and layout as much as possible. Include all tables, lists, and paragraphs. Convert the content to markdown format with appropriate headings, tables, and formatting.`;

// Pre-fill default prompt when vision is enabled
document.getElementById('vision-ingestion-enabled')?.addEventListener('change', function(e) {
    const promptField = document.getElementById('vision-prompt');
    if (promptField && e.target.checked && !promptField.value.trim()) {
        promptField.value = DEFAULT_VISION_PROMPT;
    }
});
```

#### 3.3 Update `saveWorkflowTemplate()` Function
**Location:** Lines 2117-2155

**Changes:**
1. Read vision ingestion checkbox state
2. Read vision prompt
3. Read all vision config values
4. Send `mode: 'vision'` and `vision_config` when enabled
5. Send `mode: 'programmatic'` when disabled

**Code changes:**
```javascript
// Get vision ingestion settings
const visionEnabled = document.getElementById('vision-ingestion-enabled')?.checked || false;
const visionPrompt = visionEnabled ? (document.getElementById('vision-prompt')?.value || DEFAULT_VISION_PROMPT) : null;
const visionConfig = visionEnabled ? {
    dpi: parseInt(document.getElementById('vision-dpi')?.value || '200'),
    image_format: document.getElementById('vision-image-format')?.value || 'PNG',
    jpeg_quality: parseInt(document.getElementById('vision-jpeg-quality')?.value || '85'),
    model: document.getElementById('vision-model')?.value || 'claude-3-opus-20240229',
    max_tokens: parseInt(document.getElementById('vision-max-tokens')?.value || '4096'),
    temperature: parseFloat(document.getElementById('vision-temperature')?.value || '0') / 100
} : null;

// ... existing ingestion profile code ...

// Update existing ingestion profile
const ingestionResponse = await fetch(`/api/bulk-doc-analysis/ingestion-profiles/${ingestionProfileId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        accepted_input_types: docTypes,
        mode: visionEnabled ? 'vision' : 'programmatic',  // CHANGED
        vision_prompt: visionPrompt,  // NEW
        vision_config: visionConfig  // NEW
    }),
    credentials: 'same-origin'
});

// OR create new ingestion profile
const ingestionResponse = await fetch('/api/bulk-doc-analysis/ingestion-profiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        name: name + ' Ingestion',
        accepted_input_types: docTypes,
        mode: visionEnabled ? 'vision' : 'programmatic',  // CHANGED
        vision_prompt: visionPrompt,  // NEW
        vision_config: visionConfig  // NEW
    }),
    credentials: 'same-origin'
});
```

#### 3.4 Update `editWorkflowTemplate()` Function
**Location:** Lines 2270-2300

**Changes:**
1. Fetch ingestion profile details when editing
2. Load vision mode and config
3. Populate vision form fields

**Code changes:**
```javascript
// Load ingestion profile details if we have an ID
const ingestionProfileId = fullWorkflow.ingestion_profile_id;
if (ingestionProfileId) {
    document.getElementById('workflow-template-ingestion-profile-id').value = ingestionProfileId;
    
    // Fetch full ingestion profile details
    try {
        const profileResponse = await fetch(`/api/bulk-doc-analysis/ingestion-profiles/${ingestionProfileId}`, {
            credentials: 'same-origin'
        });
        
        if (profileResponse.ok) {
            const profileData = await profileResponse.json();
            
            // Set vision ingestion enabled if mode is vision
            const visionEnabled = profileData.mode === 'vision';
            const visionCheckbox = document.getElementById('vision-ingestion-enabled');
            if (visionCheckbox) {
                visionCheckbox.checked = visionEnabled;
                // Trigger change to show/hide config
                visionCheckbox.dispatchEvent(new Event('change'));
            }
            
            // Populate vision prompt
            if (visionEnabled && profileData.vision_prompt) {
                const promptField = document.getElementById('vision-prompt');
                if (promptField) {
                    promptField.value = profileData.vision_prompt;
                }
            }
            
            // Populate vision config
            if (visionEnabled && profileData.vision_config) {
                const config = profileData.vision_config;
                
                // Set DPI
                const dpiField = document.getElementById('vision-dpi');
                if (dpiField) dpiField.value = config.dpi || 200;
                
                // Set image format
                const formatField = document.getElementById('vision-image-format');
                if (formatField) {
                    formatField.value = config.image_format || 'PNG';
                    formatField.dispatchEvent(new Event('change')); // Show/hide JPEG quality
                }
                
                // Set JPEG quality
                const jpegQualityField = document.getElementById('vision-jpeg-quality');
                if (jpegQualityField) jpegQualityField.value = config.jpeg_quality || 85;
                
                // Set model
                const modelField = document.getElementById('vision-model');
                if (modelField) modelField.value = config.model || 'claude-3-opus-20240229';
                
                // Set max tokens
                const maxTokensField = document.getElementById('vision-max-tokens');
                if (maxTokensField) maxTokensField.value = config.max_tokens || 4096;
                
                // Set temperature
                const tempField = document.getElementById('vision-temperature');
                if (tempField) {
                    tempField.value = Math.round((config.temperature || 0.0) * 100);
                    tempField.dispatchEvent(new Event('input')); // Update display
                }
            }
        }
    } catch (e) {
        console.error('Error loading ingestion profile:', e);
    }
}
```

#### 3.5 Update `resetWorkflowTemplateForm()` Function
**Location:** Lines 1809-1827

**Changes:**
- Reset vision ingestion checkbox
- Reset vision config fields
- Hide vision config container

**Code changes:**
```javascript
// Reset vision ingestion
const visionCheckbox = document.getElementById('vision-ingestion-enabled');
if (visionCheckbox) {
    visionCheckbox.checked = false;
    visionCheckbox.dispatchEvent(new Event('change')); // Hide config
}

// Reset vision fields
document.getElementById('vision-prompt')?.value = '';
document.getElementById('vision-dpi').value = '200';
document.getElementById('vision-image-format').value = 'PNG';
document.getElementById('vision-jpeg-quality').value = '85';
document.getElementById('vision-model').value = 'claude-3-opus-20240229';
document.getElementById('vision-max-tokens').value = '4096';
document.getElementById('vision-temperature').value = '0';
```

---

### 4. Remove Images from File Type Detection
**File:** `external/ai_bulk_doc_analysis/static/bulk_doc_analysis.js`

**Location:** Lines 1981-1983

**Change:**
```javascript
// REMOVE this case:
case 'Images': extensions.push('.png', '.jpg', '.jpeg'); break;
```

---

## TESTING CHECKLIST

### Backend Tests
- [ ] Create ingestion profile with vision mode and config
- [ ] Update ingestion profile vision config
- [ ] Get ingestion profile returns vision_config
- [ ] Vision ingestion processes one page at a time
- [ ] Page markers appear in output (## Page 1, ## Page 2, etc.)
- [ ] Token tracking works per page
- [ ] Error handling: failed page doesn't stop entire document
- [ ] Configurable DPI works (150, 200, 300)
- [ ] Configurable image format works (PNG, JPEG)
- [ ] JPEG quality setting works
- [ ] Different models work (Opus, Sonnet, Haiku)
- [ ] Max tokens per page works
- [ ] Temperature setting works
- [ ] Default values applied when config missing

### Frontend Tests
- [ ] Images checkbox removed
- [ ] Advanced Options section appears
- [ ] Vision ingestion checkbox toggles config visibility
- [ ] Default prompt pre-fills when enabled
- [ ] All config fields save correctly
- [ ] JPEG quality shows/hides based on format
- [ ] Temperature slider updates display
- [ ] Creating workflow with vision mode works
- [ ] Editing workflow loads vision config correctly
- [ ] Resetting form clears vision config
- [ ] Validation: prompt required when vision enabled

---

## MIGRATION NOTES

### Database Migration
**Status:** ✅ No migration needed

We're using existing `metadata_json` field, so no schema changes required.

### Backward Compatibility
- Existing ingestion profiles without `vision_config` will use defaults
- Existing vision profiles will continue to work (single API call) until updated
- Consider adding migration script to convert old vision profiles to new format

---

## IMPLEMENTATION ORDER

### Phase 1: Backend Core
1. Update `_ingest_vision()` for per-page processing
2. Add vision_config extraction logic
3. Update `create_ingestion_profile()` to accept vision_config
4. Test per-page processing with hardcoded config

### Phase 2: Backend API
5. Update API endpoints to handle vision_config
6. Test create/update/get with vision_config
7. Add default vision prompt constant

### Phase 3: Frontend UI
8. Remove Images checkbox
9. Add Advanced Options section HTML
10. Add JavaScript toggle/show/hide logic
11. Add default prompt constant

### Phase 4: Frontend Integration
12. Update `saveWorkflowTemplate()` to send vision config
13. Update `editWorkflowTemplate()` to load vision config
14. Update `resetWorkflowTemplateForm()` to clear vision config

### Phase 5: Testing & Polish
15. End-to-end testing
16. Error handling testing
17. UI/UX polish
18. Documentation updates

---

## ESTIMATED EFFORT

- **Backend:** 4-6 hours
- **Frontend:** 3-4 hours
- **Testing:** 2-3 hours
- **Total:** 9-13 hours

---

## NOTES

1. **Per-page processing benefits:**
   - Better error handling
   - Progress tracking capability
   - Avoids token limit issues
   - More resilient to timeouts

2. **Default values:**
   - DPI: 200 (good balance)
   - Format: PNG (lossless)
   - Model: Opus (best quality)
   - Max tokens: 4096 per page
   - Temperature: 0.0 (deterministic)

3. **Future enhancements:**
   - Progress bar for multi-page processing
   - Retry failed pages
   - Batch processing optimization
   - Cost estimation before processing

