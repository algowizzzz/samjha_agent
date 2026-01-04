# How to Create Chains and Workflows

**Date:** 2025-01-27  
**Status:** Current Backend Support Summary

## Summary

### ✅ **What IS Supported:**

1. **Workflow API Endpoints** (Full CRUD via HTTP):
   - `POST /api/bulk-doc-analysis/workflows` - Create workflow
   - `GET /api/bulk-doc-analysis/workflows` - List workflows
   - `GET /api/bulk-doc-analysis/workflows/<id>` - Get workflow
   - `PUT /api/bulk-doc-analysis/workflows/<id>` - Update workflow (creates new version)
   - `DELETE /api/bulk-doc-analysis/workflows/<id>` - Delete workflow

2. **Chain Service Methods** (Programmatic only, no HTTP endpoints):
   - `BulkDocDBService.create_chain()` - Create chain (Python only)
   - `BulkDocDBService.update_chain()` - Update chain (Python only)
   - `BulkDocDBService.get_chain()` - Get chain (Python only)

### ❌ **What is NOT Supported:**

1. **Chain API Endpoints** - Missing:
   - `POST /api/bulk-doc-analysis/chains` - ❌ **404 Error**
   - `GET /api/bulk-doc-analysis/chains` - ❌ Not implemented
   - `PUT /api/bulk-doc-analysis/chains/<id>` - ❌ Not implemented
   - `DELETE /api/bulk-doc-analysis/chains/<id>` - ❌ Not implemented

2. **UI Chain Creation** - Broken:
   - The "Create New Chain" button in `/bulk-doc-analysis` page tries to call the missing API endpoint
   - Results in 404 error

---

## How to Create a Chain

### **Option 1: Programmatic (Python) - RECOMMENDED**

Use the `BulkDocDBService.create_chain()` method directly:

```python
from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db

# Initialize database
init_db()

# Create service instance
svc = BulkDocDBService()

# Create chain
chain = svc.create_chain(
    user_id="admin",
    name="Simple Obligation Extractor",
    description="Extract obligations from documents",
    steps=[
        {
            "index": 1,
            "title": "Extract Obligations",
            "prompt": "Extract all key requirements...",
            "description": "Extract all explicit requirement statements",
            "required_inputs": ["R0"],
            "model_config": {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 4096,
                "temperature": 0.2
            }
        }
    ]
)

# Get chain_version_id for use in workflow
chain_version_id = chain.chain_version_id
print(f"Created chain version: {chain_version_id}")
```

**Example:** See `test_workflow.py` lines 123-180 for a complete example.

### **Option 2: Fix Missing API Endpoints (Future)**

The chain service exists, but HTTP endpoints need to be added to `blueprint.py`. This would require:
1. Adding `@bp.route("/api/bulk-doc-analysis/chains", methods=["POST"])` endpoint
2. Adding GET, PUT, DELETE endpoints for chains
3. Implementing authentication and validation

---

## How to Create a Workflow

### **Via API (HTTP) - FULLY SUPPORTED** ✅

```bash
POST /api/bulk-doc-analysis/workflows
Content-Type: application/json

{
  "name": "My Workflow",
  "description": "A workflow that processes documents (20-240 chars required)",
  "domains": ["finance", "risk"],
  "ingestion_profile_id": "ing_abc123...",
  "chain_version_id": "cv_chain_xyz-v1",  # From chain creation
  "export_profile_id": "exp_def456..."
}
```

**Required Fields:**
- `name`: 3-80 characters
- `description`: 20-240 characters
- `domains`: Array of at least 1 domain name
- `ingestion_profile_id`: Must exist
- `chain_version_id`: Must exist (from chain creation)
- `export_profile_id`: Must exist

**Response:**
```json
{
  "success": true,
  "workflow": {
    "workflow_id": "wf_abc123...",
    "name": "My Workflow",
    "description": "..."
  }
}
```

### **Via Python (Programmatic)**

```python
from external.ai_bulk_doc_analysis.workflow_service import WorkflowService

workflow_service = WorkflowService()

workflow = workflow_service.create_workflow(
    user_id="admin",
    name="My Workflow",
    description="A workflow that processes documents",
    domains=["finance", "risk"],
    ingestion_profile_id="ing_abc123...",
    chain_version_id="cv_chain_xyz-v1",
    export_profile_id="exp_def456..."
)
```

---

## Complete Workflow Creation Process

### **Step-by-Step:**

1. **Create Ingestion Profile** (via API or Python):
   ```python
   # See test_workflow.py:61-91
   from external.ai_bulk_doc_analysis.ingestion_service import IngestionService
   from external.ai_bulk_doc_analysis.db_service import init_db, get_db_session
   from external.ai_bulk_doc_analysis.models import IngestionProfile
   import uuid
   
   init_db()
   with get_db_session() as db:
       ingestion_profile_id = f"ing_{uuid.uuid4().hex[:12]}"
       profile = IngestionProfile(
           ingestion_profile_id=ingestion_profile_id,
           name="PDF Programmatic Ingestion",
           accepted_input_types=["PDF"],
           mode="programmatic",
           vision_prompt=None
       )
       db.add(profile)
       db.commit()
   ```

2. **Create Export Profile** (via API or Python):
   ```python
   # See test_workflow.py:93-121
   from external.ai_bulk_doc_analysis.export_service import ExportService
   from external.ai_bulk_doc_analysis.db_service import init_db, get_db_session
   from external.ai_bulk_doc_analysis.models import ExportProfile
   import uuid
   
   init_db()
   with get_db_session() as db:
       export_profile_id = f"exp_{uuid.uuid4().hex[:12]}"
       profile = ExportProfile(
           export_profile_id=export_profile_id,
           name="Markdown Export",
           format="MD",
           config_json={}
       )
       db.add(profile)
       db.commit()
   ```

3. **Create Chain** (PROGRAMMATIC ONLY):
   ```python
   # See test_workflow.py:123-180
   from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db
   
   init_db()
   svc = BulkDocDBService()
   
   chain = svc.create_chain(
       user_id="admin",
       name="Simple Obligation Extractor",
       description="Extract obligations from documents",
       steps=[{...}]  # Step definitions
   )
   chain_version_id = chain.chain_version_id
   ```

4. **Create Workflow** (via API):
   ```bash
   POST /api/bulk-doc-analysis/workflows
   {
     "name": "My Workflow",
     "description": "Process documents and extract obligations",
     "domains": ["finance"],
     "ingestion_profile_id": "...",
     "chain_version_id": "...",  # From step 3
     "export_profile_id": "..."
   }
   ```

---

## Why Chains Have No API Endpoints

**Historical Context:**
- Comment in `blueprint.py:921`: "Chain APIs (DEPRECATED - Remove in Phase 1)"
- Chains were marked for deprecation in favor of workflows
- However, chains are still required as components of workflows
- The deprecation was premature - chains are still needed

**Current State:**
- Chain service methods exist and work (`db_service.py`)
- HTTP endpoints were removed/never implemented
- UI still tries to use chain endpoints (causing 404 errors)
- Workflows cannot be created without chains

---

## Recommendations

### **Short-term (Immediate):**
1. Use programmatic chain creation (`BulkDocDBService.create_chain()`) when needed
2. Create workflows via API once you have a `chain_version_id`
3. See `test_workflow.py` for a complete working example

### **Long-term (Fix UI):**
1. **Option A**: Implement missing chain API endpoints in `blueprint.py`
   - Add POST, GET, PUT, DELETE endpoints for chains
   - Restore UI functionality

2. **Option B**: Update UI to create workflows directly
   - Modify UI to create workflows instead of chains
   - Workflows can then reference chain versions
   - More complex UI change

3. **Option C**: Hybrid approach
   - Add chain endpoints for backward compatibility
   - Encourage workflow usage going forward
   - Best of both worlds

---

## Example: Complete Script

See `test_workflow.py` for a complete working example that:
1. Creates ingestion profile
2. Creates export profile  
3. Creates chain (programmatic)
4. Creates workflow (programmatic)
5. Uploads document
6. Creates and runs workflow execution

This script demonstrates the current recommended approach for creating chains and workflows.

