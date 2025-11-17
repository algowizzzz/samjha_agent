# Phase 3 - Template Mapping & Refinement (LLM-driven)

## Overview
Phase 3 uses LLM to map document sections to a predefined template, improve titles, score relevancy, and perform a second-pass mapping of high-relevance unmapped sections.

## Prerequisites
- Phase 1 completed (provides: `raw_markdown`, `heading_structure`, `file_metadata`)
- Phase 2 completed (provides: `chunks`, `index.sections`, `template`)
- LLM configured and available

## Phase 3 Steps

### 1. `heading_mapping_llm` → Maps sections to template

**Purpose**: Maps document sections to template sections using LLM

**Inputs**:
- `state.template` - Template configuration with outline structure
- `state.index.sections` - List of document sections from Phase 2
  - Each section contains: `file_section_id`, `heading_text_original`, `heading_path`

**LLM Input Payload**:
```json
{
  "template": {
    "sections": [...],  // Template outline structure
    ...
  },
  "sections": [
    {
      "file_section_id": "section_001",
      "heading_text": "Introduction",
      "heading_path": ["Chapter 1", "Introduction"]
    },
    ...
  ]
}
```

**LLM System Prompt**:
> "You map document sections to a predefined outline template. For each section, return JSON objects with file_section_id, mapped (bool), optional template_section_id, template_path (array of titles), and confidence (0-1)."

**Expected LLM Output**:
```json
[
  {
    "file_section_id": "section_001",
    "mapped": true,
    "template_section_id": "intro",
    "template_path": ["Overview", "Introduction"],
    "confidence": 0.95
  },
  {
    "file_section_id": "section_002",
    "mapped": false,
    "template_section_id": null,
    "template_path": null,
    "confidence": 0.2
  },
  ...
]
```

**Outputs/Updates**:
- Updates `state.index.sections[i].template_mapping`:
  - `mapped`: boolean
  - `template_section_id`: string or null
  - `template_path`: array of strings or null
  - `confidence`: float (0-1)
- Updates `state.index.sections[i].status`:
  - `"mapped"` if mapped = true
  - `"unmapped"` if mapped = false
- Logs: `heading_mapping_llm: mapped={count}`

**Expected Results** (for CAR_Chapter_1_Overview):
- Some sections mapped to template (e.g., introduction, definitions)
- Some sections unmapped (not in template structure)
- Confidence scores vary (0.0-1.0)

---

### 2. `title_improver_llm` → Improves section titles

**Purpose**: Rewrites mapped section titles for clarity and professionalism

**Inputs**:
- `state.index.sections` - Only sections where `status == "mapped"`

**LLM Input Payload**:
```json
{
  "sections": [
    {
      "file_section_id": "section_001",
      "heading_text_original": "Intro",
      "template_path": ["Overview", "Introduction"]
    },
    ...
  ]
}
```

**LLM System Prompt**:
> "You rewrite section titles for clarity and professionalism. Return JSON with file_section_id and heading_text_improved. Use same language as input."

**Expected LLM Output**:
```json
[
  {
    "file_section_id": "section_001",
    "heading_text_improved": "Introduction to Capital Adequacy"
  },
  {
    "file_section_id": "section_002",
    "heading_text_improved": "Regulatory Framework Overview"
  },
  ...
]
```

**Outputs/Updates**:
- Updates `state.index.sections[i].heading_text_improved` for mapped sections
- Logs: `title_improver_llm: improved={count}`

**Expected Results**:
- Mapped sections get improved titles
- Titles maintain original language
- Titles are more professional and clear
- Count of improved titles equals count of mapped sections

---

### 3. `relevancy_scoring_llm` → Scores unmapped sections

**Purpose**: Rates how relevant unmapped sections are to the overall document outline

**Inputs**:
- `state.index.sections` - Only sections where `status == "unmapped"`
- `state.chunks` - Chunk data for content excerpts

**LLM Input Payload**:
```json
{
  "sections": [
    {
      "file_section_id": "section_005",
      "heading_text": "Appendix A",
      "content_excerpt": "This section contains supplementary data..."  // First 1000 chars
    },
    ...
  ]
}
```

**LLM System Prompt**:
> "You rate how relevant an unmapped document section is to the overall outline. Return file_section_id, relevancy_score (0-1), and category from [maybe_append, low_value, append_anyway]."

**Expected LLM Output**:
```json
[
  {
    "file_section_id": "section_005",
    "relevancy_score": 0.75,
    "category": "maybe_append"
  },
  {
    "file_section_id": "section_010",
    "relevancy_score": 0.15,
    "category": "low_value"
  },
  ...
]
```

**Outputs/Updates**:
- Updates `state.index.sections[i].relevancy_score_if_unmapped`: float (0-1)
- Updates `state.index.sections[i].template_mapping.relevancy_score`: float
- Updates `state.index.sections[i].template_mapping.category`: string
  - One of: `"maybe_append"`, `"low_value"`, `"append_anyway"`
- Logs: `relevancy_scoring_llm: scored={count}`

**Expected Results**:
- Unmapped sections receive relevancy scores (0.0-1.0)
- Categories assigned based on relevance
- High-relevance sections marked for potential second-pass mapping

---

### 4. `second_pass_mapping_llm` → Maps high-relevance unmapped sections

**Purpose**: Second pass to map relevant but previously unmapped sections to remaining template sections

**Inputs**:
- `state.template` - Template configuration
- `state.index.sections` - Unmapped sections with `relevancy_score_if_unmapped >= threshold`
  - Default threshold: 0.4 (configurable via `config.template.late_append_relevancy_threshold`)

**LLM Input Payload**:
```json
{
  "template": {...},
  "sections": [
    {
      "file_section_id": "section_005",
      "heading_text": "Appendix A",
      "relevancy_score": 0.75
    },
    ...
  ]
}
```

**LLM System Prompt**:
> "You receive relevant but previously unmapped sections. Decide if they can map to remaining template sections. Return file_section_id, mapped (bool), optional template_section_id, and status (appended_late or unmapped)."

**Expected LLM Output**:
```json
[
  {
    "file_section_id": "section_005",
    "mapped": true,
    "template_section_id": "appendix",
    "status": "appended_late"
  },
  {
    "file_section_id": "section_010",
    "mapped": false,
    "template_section_id": null,
    "status": "unmapped"
  },
  ...
]
```

**Outputs/Updates**:
- Updates `state.index.sections[i].template_mapping`:
  - `mapped`: true (if mapped in second pass)
  - `template_section_id`: string or null
- Updates `state.index.sections[i].status`:
  - `"appended_late"` if mapped = true
  - `"unmapped"` if mapped = false (remains unmapped)
- Logs: `second_pass_mapping_llm: appended={count}`

**Expected Results**:
- Some high-relevance unmapped sections get mapped in second pass
- Status changes from `"unmapped"` to `"appended_late"` for newly mapped sections
- Lower-relevance sections remain unmapped

---

### 5. `_recompute_index_stats` → Recomputes index statistics

**Purpose**: Updates index statistics after Phase 3 mappings

**Outputs/Updates**:
- Updates `state.index.stats` with:
  - `total_sections`: total count
  - `mapped_sections`: count of mapped sections
  - `unmapped_sections`: count of unmapped sections
  - `appended_late_sections`: count of late-appended sections
  - `mapping_confidence_avg`: average confidence score

---

## Summary of Phase 3 State Changes

### Input State (from Phase 2):
```json
{
  "index": {
    "sections": [
      {
        "file_section_id": "section_001",
        "heading_text_original": "Introduction",
        "status": null,
        "template_mapping": {}
      },
      ...
    ]
  },
  "template": {...}
}
```

### Output State (after Phase 3):
```json
{
  "index": {
    "sections": [
      {
        "file_section_id": "section_001",
        "heading_text_original": "Introduction",
        "heading_text_improved": "Introduction to Capital Adequacy",  // Added
        "status": "mapped",  // Updated
        "template_mapping": {  // Updated
          "mapped": true,
          "template_section_id": "intro",
          "template_path": ["Overview", "Introduction"],
          "confidence": 0.95
        }
      },
      {
        "file_section_id": "section_005",
        "heading_text_original": "Appendix A",
        "status": "appended_late",  // Updated
        "relevancy_score_if_unmapped": 0.75,  // Added
        "template_mapping": {  // Updated
          "mapped": true,
          "template_section_id": "appendix",
          "relevancy_score": 0.75,
          "category": "maybe_append"
        }
      },
      {
        "file_section_id": "section_010",
        "heading_text_original": "References",
        "status": "unmapped",  // Updated
        "relevancy_score_if_unmapped": 0.15,  // Added
        "template_mapping": {  // Updated
          "mapped": false,
          "relevancy_score": 0.15,
          "category": "low_value"
        }
      },
      ...
    ],
    "stats": {  // Updated
      "total_sections": 10,
      "mapped_sections": 6,
      "unmapped_sections": 3,
      "appended_late_sections": 1,
      "mapping_confidence_avg": 0.82
    }
  }
}
```

## Expected Results Summary

After Phase 3 completion for a typical document (e.g., CAR_Chapter_1_Overview):

1. **Mapping Results**:
   - ~60-80% of sections mapped to template
   - ~20-40% of sections remain unmapped initially
   - ~5-15% of unmapped sections mapped in second pass

2. **Title Improvements**:
   - All mapped sections have improved titles
   - Titles are clearer and more professional

3. **Relevancy Scores**:
   - All unmapped sections have relevancy scores (0.0-1.0)
   - Categories assigned: `maybe_append`, `low_value`, or `append_anyway`

4. **Index Statistics**:
   - Total sections count
   - Mapped vs unmapped counts
   - Average confidence scores

## Notes

- **LLM Dependencies**: All Phase 3 nodes require LLM to be configured and available
- **Error Handling**: If LLM unavailable, nodes skip gracefully with log messages
- **Configurable**: Second-pass threshold configurable via `config.template.late_append_relevancy_threshold`
- **Non-Destructive**: Original headings preserved in `heading_text_original`, improvements in `heading_text_improved`

