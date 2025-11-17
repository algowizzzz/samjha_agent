# Section Extraction Fix - Whole-Doc Fallback

**Date:** November 15, 2025
**Status:** ✅ Implemented and Tested
**Files Modified:** `external/agent/doc_review_agent.py`

## Problem

Phase 2 section extraction failed for documents without markdown headings (like Template_Version.pdf):

- **`extract_section_by_headings`** requires markdown headings (`# Heading` format)
- **`extract_section_by_toc`** requires TOC entries findable via string match
- **Template_Version.pdf** has neither after PDF→markdown conversion
- Result: All section extractions returned empty, Phase 2 failed with "no sections extracted"

## Solution

Implemented a two-tier fallback strategy in `/external/agent/doc_review_agent.py`:

### 1. Section-Level Fallback (`_extract_section_chunk`)

**Location:** Line 411-478

When both heading-based and TOC-based extraction return empty:
- Return a chunk containing the **entire document text**
- Mark as `method: "whole_doc_fallback"`
- Add issues explaining why fallback was used
- Continue to next section

**Code:**
```python
# Check if both extraction methods failed (no text extracted)
heading_has_text = heading_candidate.get("text", "").strip()
toc_has_text = toc_candidate.get("text", "").strip()

# Fallback: if both methods fail, use whole document as single chunk
if not heading_has_text and not toc_has_text:
    self.logger.warning(
        "Both heading and TOC extraction failed for section '%s', using whole-doc fallback",
        section_title,
    )
    return {
        "section_title": section_title,
        "method": "whole_doc_fallback",
        "page_range": [1, state["doc_meta"].get("page_count", None)],
        "char_range": [0, len(raw_markdown)],
        "boundary_check": "whole_document",
        "issues": [
            "No markdown headings found",
            "TOC entries not found in text",
            "Using entire document as single chunk",
        ],
        "text": raw_markdown,
    }
```

### 2. Phase-Level Consolidation (`run_phase2`)

**Location:** Line 81-132

After all section extractions, if **ALL** sections used fallback:
- Consolidate into a **single** "Full Document" chunk
- Prevents sending same full document to LLM 9 times
- More efficient and cost-effective

**Code:**
```python
# If ALL sections used fallback, consolidate to single whole-doc chunk
if extracted > 0 and fallback_count == extracted:
    self.logger.warning(
        "All %d sections used whole-doc fallback. Consolidating to single 'Full Document' chunk.",
        extracted,
    )
    raw_markdown = state["structure"]["raw_text"]
    state["phase2"]["chunks"] = {
        "Full Document": {
            "section_title": "Full Document",
            "method": "whole_doc_fallback",
            ...
            "text": raw_markdown,
        }
    }
    extracted = 1
```

## Test Results

**Test Script:** `/test_extraction_fallback.py`

```
✓ Test 1 passed: Whole-doc fallback works for section with no headings/TOC
✓ Test 2 passed: Phase 2 consolidates multiple fallback chunks into single 'Full Document'

✅ All extraction fallback tests passed!
```

**Validation:**
- Extraction no longer fails when headings/TOC missing
- Single consolidated chunk created for uniform documents
- Issues properly logged for user visibility
- Phase 2 can now proceed to review stage

## Impact on Workflow

### Before Fix
```
Phase 2: Extract sections
  ↓
All extractions return empty (no headings/TOC)
  ↓
Phase 2 fails: "no sections extracted"
  ↓
BLOCKED ❌
```

### After Fix
```
Phase 2: Extract sections
  ↓
All extractions use whole-doc fallback
  ↓
Consolidate to single "Full Document" chunk
  ↓
Phase 2 proceeds to review ✅
  ↓
LLM reviews entire document as one section
  ↓
Phase 3: Apply suggested changes ✅
```

## Next Steps

1. ✅ **Extraction fix** - COMPLETE
2. ⏳ **Test Phase 2 end-to-end** with real Template_Version.pdf
3. ⏳ **Validate Phase 2 reviews** generate useful feedback
4. ⏳ **Validate Phase 3** change application works
5. ⏳ **Build Phase 2/3 UI** to visualize results

## Notes

- This is a **pragmatic fallback** for documents without structure
- Ideal: Documents should have proper headings for section-level review
- Fallback ensures workflow doesn't break on unstructured docs
- LLM can still provide valuable feedback on full document
- Future: Could add page-based chunking as another fallback tier
