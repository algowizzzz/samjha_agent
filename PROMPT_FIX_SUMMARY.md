# Prompt Fix Summary - ETL Multimodal Header Issue

## Problem Identified

During the initial ETL multimodal extraction phase, markdown header symbols (`#`, `##`, `###`) were being **added** to the original content, destroying the integrity of the source document.

### Root Cause

In `/external/tools/doc_processing/convert_to_markdown.py`, the vision transcription prompt was explicitly instructing the LLM to:
- Add `#` symbols to text that **looked like** headings (large, bold, standalone)
- Interpret visual formatting and convert it to markdown syntax
- Transform "Document title (largest)" → `# Title`
- Transform "Section heading (large, bold)" → `## 1. Overview`

**This was destructive because:**
- Original PDF content doesn't contain `#`, `##`, `###` symbols
- The LLM was adding markup that wasn't in the source
- This corrupted the original content for downstream processing

## Files Modified

### 1. `/external/tools/doc_processing/convert_to_markdown.py`

**Three functions updated:**

#### A. `_transcribe_page_with_vision()` (lines 103-146)
**Before:** Instructed to add markdown headers based on visual cues
**After:** Instructs to transcribe EXACTLY as written, NO markdown interpretation

Key prompt changes:
```diff
- 3. Use # ## ### ONLY for text that is VISUALLY a heading
+ 3. DO NOT ADD markdown symbols (#, ##, ###) - transcribe text EXACTLY as it appears
+ 4. DO NOT interpret headings - just transcribe the text literally
+ 5. Preserve ALL original text character-for-character

- HEADINGS (large, bold, standalone):
- - Document title (largest) → # Title
- - Section heading (large, bold) → ## 1. Overview
+ TRANSCRIPTION RULES:
+ - Large/bold text → transcribe the text as-is (no # symbols)
```

#### B. `_create_semantic_blocks_with_llm()` (lines 217-268)
**Before:** Expected markdown with `#` symbols, used types like `heading1`, `heading2`, `heading3`
**After:** Works with literal text, detects headings by context (short lines, title case, standalone)

Key changes:
```diff
- Analyze this markdown and group it into semantic blocks
+ Analyze this text and group it into semantic blocks

+ - Detect headings by context (short lines, title case, standalone) NOT by # symbols
+ - The text does NOT contain # symbols - it's literal transcription

- BLOCK TYPES:
- - heading1, heading2, heading3
+ BLOCK TYPES:
+ - heading (short, standalone lines that appear to be titles/headings)

+ HOW TO DETECT HEADINGS (without # symbols):
+ - Short lines (typically < 60 chars)
+ - Title case or ALL CAPS
+ - Standalone (not part of paragraph)
```

#### C. `_detect_block_type()` (lines 188-222)
**Before:** Detected headings by checking for `#`, `##`, `###` prefixes
**After:** Detects headings by analyzing content structure (length, case, standalone)

Key changes:
```diff
- if first_line.startswith('### '): return 'heading3'
- if first_line.startswith('## '): return 'heading2'
- if first_line.startswith('# '): return 'heading1'

+ # Heading detection (without # symbols):
+ # Short lines (< 80 chars), single line, title-like
+ lines = stripped.split('\n')
+ if len(lines) == 1 and len(first_line) < 80:
+     words = first_line.split()
+     if words and (first_line.isupper() or sum(w[0].isupper() for w in words if w) > len(words) * 0.5):
+         return 'heading'
```

## Testing Results

✅ Block type detection test passed:
- `"Risk Management Guideline"` → `heading` (detected without # symbols)
- `"COMPLIANCE POLICY"` → `heading` (detected by ALL CAPS)
- Normal paragraphs → `paragraph`
- Bullet lists → `bullet`
- Numbered lists → `numbered`
- Tables → `table`
- Empty lines → `empty`

## Impact

### What's Fixed ✅
- Vision extraction now preserves original content character-for-character
- No markdown headers added during transcription
- Block type detection works with literal text
- Semantic block grouping works with context-based heading detection

### What's Preserved ✅
- Table of Contents generation still works (intentionally adds "## Table of Contents" header)
- Block metadata tracking still functional
- Verification and suggestion logic unchanged
- UI annotation and display logic unchanged

### Downstream Benefits 📊
- Original content integrity maintained throughout pipeline
- Phase 2 section reviews work with actual content
- Phase 3 change tracking accurate to source
- UI can display true source content without corruption

## Verification Needed

To fully test the fix, run an end-to-end document review with a PDF:
```bash
# Process a PDF document and verify no # symbols added to content
python test_complete_doc_review_api.py
```

Check that:
1. Extracted markdown has NO added `#` symbols
2. Headings are detected and tracked in metadata
3. UI correctly displays original content
4. Block IDs and semantic grouping still work

