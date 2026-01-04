# Workflow Creation - Required Inputs Guide

**Date:** 2025-01-27

## Quick Answer

**YES - You MUST create a Chain first before creating a Workflow.**

**Where do prompts go?** → **In the Chain** (not directly in the workflow)

---

## Workflow Architecture

```
Workflow
  ├─ Ingestion Profile (how to read input files: PDF, DOCX, CSV, etc.)
  ├─ Chain Version (contains prompts/steps - THIS IS WHERE YOUR PROMPTS GO)
  └─ Export Profile (output format: MD, JSON, CSV, DOCX, PDF)
```

**Key Point:** Prompts/steps are stored in **Chains**, and workflows reference chains via `chain_version_id`.

---

## Required Inputs for Workflow Creation

### 1. **Workflow Metadata** (Direct inputs)

```json
{
  "name": "My Workflow Name",              // Required, 3-80 characters
  "description": "Workflow description...", // Required, 20-240 characters
  "domains": ["finance", "risk"]            // Required, array with at least 1 domain
}
```

### 2. **Ingestion Profile ID** (Reference)

**What it is:** Defines how documents are ingested/converted
- Input types: PDF, DOCX, TXT, MD, PNG, CSV
- Conversion mode: `programmatic`, `vision`
- Vision prompt (if using vision mode)

**How to create:**
```python
# Via API
POST /api/bulk-doc-analysis/ingestion-profiles
{
  "name": "PDF Ingestion",
  "accepted_input_types": ["PDF"],
  "mode": "programmatic"
}
```

### 3. **Chain Version ID** (Reference - **THIS CONTAINS YOUR PROMPTS**)

**What it is:** Contains the AI processing steps (prompts)
- Steps array with prompts, model config, required inputs
- This is where you define what the AI does with documents

**How to create (programmatic only):**
```python
from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db

init_db()
svc = BulkDocDBService()

chain = svc.create_chain(
    user_id="admin",
    name="My Chain",
    description="Chain description",
    steps=[
        {
            "index": 1,
            "title": "Step 1 Title",
            "prompt": "YOUR PROMPT HERE - This is where prompts go!",
            "description": "Step description",
            "required_inputs": ["R0"],  # R0 = raw document content
            "model_config": {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 4096,
                "temperature": 0.2
            }
        },
        {
            "index": 2,
            "title": "Step 2 Title",
            "prompt": "Another prompt using R0 or R1 (previous step output)",
            "description": "Step 2 description",
            "required_inputs": ["R0", "R1"],  # Can reference previous steps
            "model_config": {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 8192,
                "temperature": 0.3
            }
        }
    ]
)

chain_version_id = chain.chain_version_id  # Use this in workflow
```

### 4. **Export Profile ID** (Reference)

**What it is:** Defines output format
- Formats: MD, JSON, CSV, DOCX, PDF
- Format-specific configuration

**How to create:**
```python
# Via API
POST /api/bulk-doc-analysis/export-profiles
{
  "name": "Markdown Export",
  "format": "MD",
  "config_json": {}
}
```

---

## Complete Workflow Creation Flow

### Step 1: Create Ingestion Profile

```bash
POST /api/bulk-doc-analysis/ingestion-profiles
Content-Type: application/json

{
  "name": "PDF Programmatic",
  "accepted_input_types": ["PDF"],
  "mode": "programmatic",
  "vision_prompt": null
}
```

**Response:**
```json
{
  "success": true,
  "profile": {
    "ingestion_profile_id": "ing_abc123..."
  }
}
```

### Step 2: Create Export Profile

```bash
POST /api/bulk-doc-analysis/export-profiles
Content-Type: application/json

{
  "name": "Markdown Export",
  "format": "MD",
  "config_json": {}
}
```

**Response:**
```json
{
  "success": true,
  "profile": {
    "export_profile_id": "exp_def456..."
  }
}
```

### Step 3: Create Chain (PROGRAMMATIC - Contains Your Prompts)

```python
from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db

init_db()
svc = BulkDocDBService()

# YOUR PROMPTS GO HERE IN THE STEPS ARRAY
chain = svc.create_chain(
    user_id="admin",
    name="Obligation Extractor Chain",
    description="Extract obligations from documents",
    steps=[
        {
            "index": 1,
            "title": "Extract Obligations",
            "prompt": """Extract all key requirements and obligations from the following document.

Document content:
{R0}

Output a JSON array with each requirement containing:
- type: MUST | SHOULD | MAY | OTHER
- statement: the exact requirement text
- section: section heading

Output valid JSON only.""",
            "description": "Extract obligations from document",
            "required_inputs": ["R0"],  # R0 = raw document content
            "model_config": {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 4096,
                "temperature": 0.2
            }
        }
    ]
)

print(f"Chain version ID: {chain.chain_version_id}")
```

### Step 4: Create Workflow (References Chain)

```bash
POST /api/bulk-doc-analysis/workflows
Content-Type: application/json

{
  "name": "Document Obligation Extractor",
  "description": "Extract obligations from PDF documents and output as Markdown",
  "domains": ["finance"],
  "ingestion_profile_id": "ing_abc123...",
  "chain_version_id": "cv_chain_xyz-v1",  # From Step 3
  "export_profile_id": "exp_def456..."
}
```

---

## Understanding Step Inputs (R0, R1, R2, etc.)

**Step inputs reference previous outputs:**

- `R0` = Raw document content (from ingestion)
- `R1` = Output from step 1
- `R2` = Output from step 2
- `R3` = Output from step 3
- etc.

**Example Multi-Step Chain:**

```python
steps=[
    {
        "index": 1,
        "prompt": "Extract obligations from: {R0}",
        "required_inputs": ["R0"]  # Uses raw document
    },
    {
        "index": 2,
        "prompt": "Summarize these obligations: {R1}",
        "required_inputs": ["R1"]  # Uses step 1 output
    },
    {
        "index": 3,
        "prompt": "Combine original doc {R0} with summary {R2}",
        "required_inputs": ["R0", "R2"]  # Can use multiple inputs
    }
]
```

---

## Complete Example: Creating a Simple One-Step Workflow

```python
#!/usr/bin/env python3
"""Complete example: Create a workflow with prompts"""

from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db, get_db_session
from external.ai_bulk_doc_analysis.models import IngestionProfile, ExportProfile
from external.ai_bulk_doc_analysis.workflow_service import WorkflowService
import uuid
import requests

BASE_URL = "http://localhost:8000"

# 1. Create Ingestion Profile (via API)
response = requests.post(f"{BASE_URL}/api/bulk-doc-analysis/ingestion-profiles", json={
    "name": "PDF Ingestion",
    "accepted_input_types": ["PDF"],
    "mode": "programmatic"
})
ingestion_profile_id = response.json()["profile"]["ingestion_profile_id"]

# 2. Create Export Profile (via API)
response = requests.post(f"{BASE_URL}/api/bulk-doc-analysis/export-profiles", json={
    "name": "Markdown Export",
    "format": "MD",
    "config_json": {}
})
export_profile_id = response.json()["profile"]["export_profile_id"]

# 3. Create Chain with Prompts (PROGRAMMATIC)
init_db()
svc = BulkDocDBService()

chain = svc.create_chain(
    user_id="admin",
    name="Simple Extractor",
    description="Extract key information",
    steps=[
        {
            "index": 1,
            "title": "Extract Information",
            "prompt": """Extract key information from this document:

{R0}

Output a JSON object with:
- summary: brief summary
- key_points: array of key points
- action_items: array of action items

Output valid JSON only.""",
            "description": "Extract key information",
            "required_inputs": ["R0"],
            "model_config": {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 4096,
                "temperature": 0.2
            }
        }
    ]
)
chain_version_id = chain.chain_version_id

# 4. Create Workflow (via API)
response = requests.post(f"{BASE_URL}/api/bulk-doc-analysis/workflows", json={
    "name": "Information Extractor Workflow",
    "description": "Extract key information from PDF documents and export as Markdown",
    "domains": ["general"],
    "ingestion_profile_id": ingestion_profile_id,
    "chain_version_id": chain_version_id,
    "export_profile_id": export_profile_id
})

workflow_id = response.json()["workflow"]["workflow_id"]
print(f"Created workflow: {workflow_id}")
```

---

## Summary

| Input | Where It Goes | How to Create | Contains Prompts? |
|-------|--------------|---------------|-------------------|
| **Workflow Metadata** | Direct in workflow | Via API or Python | ❌ No |
| **Ingestion Profile** | Referenced by ID | Via API | ❌ No (conversion settings) |
| **Chain** | Referenced by version ID | **Programmatic only** | ✅ **YES - Prompts go here!** |
| **Export Profile** | Referenced by ID | Via API | ❌ No (output format) |

**Key Takeaways:**
1. ✅ **Chains are REQUIRED** - workflows cannot be created without a chain
2. ✅ **Prompts go IN chains**, not in workflows
3. ✅ **Chains must be created programmatically** (no API endpoint currently)
4. ✅ Workflow just combines: Ingestion → Chain (with prompts) → Export

