# Build Details: Workflow System Implementation

**Based on:** Current chain editor in admin panel + Requirements from `enhancements_prod.md`  
**Date:** 2025-01-XX

---

## 1. Current Implementation Analysis

### 1.1 Current Chain Editor (Admin Panel)

**Location:** `web/templates/admin.html` (lines ~1100-1632)

**Current Features:**
- Modal-based editor (`chain-editor-modal`)
- Chain name (required)
- Chain description (optional)
- Steps with:
  - Required inputs (checkboxes: R0, R1, R2...)
  - Model config (model, max_tokens, temperature)
  - Prompt (textarea, required)
  - Description (optional)
- Create/Edit modes
- Validation (name required, at least 1 step, each step needs prompt)

**Current API Endpoints:**
- `GET /api/bulk-doc-analysis/chains` - List chains
- `POST /api/bulk-doc-analysis/chains` - Create chain
- `PUT /api/bulk-doc-analysis/chains/<chain_id>` - Update chain

**Current Data Structure:**
```javascript
chainEditorState = {
    mode: 'create' | 'edit',
    chainId: string | null,
    name: string,
    description: string,
    steps: [{
        index: number,
        required_inputs: string[],  // ['R0', 'R1', ...]
        prompt: string,
        description: string,
        model_config: {
            model: string,
            max_tokens: number,
            temperature: number
        }
    }]
}
```

**Current Backend:**
- `BulkDocService.create_chain()` - Creates chain with version
- `BulkDocService.update_chain()` - Updates in-place (no versioning yet)
- Validation: `_validate_chain_structure()`

---

## 2. What Needs to Change

### 2.1 Database Schema Changes

#### New Tables (See IMPLEMENTATION_ANALYSIS.md Section 1.1)

**Key Points:**
- `workflows` table (top-level entity)
- `workflow_versions` (immutable snapshots)
- `workflow_domains` (many-to-many)
- `ingestion_profiles` (with `vision_prompt` TEXT column, not file path)
- `export_profiles`
- `execution_tasks` (for CSV row-based execution)

#### Schema Modifications

**`chain_steps` table:**
```sql
ALTER TABLE chain_steps ADD COLUMN title VARCHAR(500) NOT NULL DEFAULT 'Untitled Step';
-- After migration, update existing rows, then make NOT NULL
```

**`runs` table:**
```sql
ALTER TABLE runs ADD COLUMN workflow_version_id VARCHAR(255) REFERENCES workflow_versions(workflow_version_id);
-- Initially nullable, then make required
```

**`step_results` table:**
```sql
ALTER TABLE step_results ADD COLUMN task_id VARCHAR(255) REFERENCES execution_tasks(task_id);
-- For CSV: task_id populated, for non-CSV: NULL
```

### 2.2 UI Changes Required

#### A) Admin Panel: Workflow Builder (New Section)

**Location:** Add new section in `admin.html` (similar to "Unstructured Workflows" section)

**Structure:**
```
Admin Panel
├── Existing sections (Agents, Tools, Users, Prompts)
└── NEW: "Workflow Builder" section
    ├── List of workflows (cards/rows)
    ├── "Create New Workflow" button
    └── Workflow editor modal
```

**Workflow Editor Modal Structure:**
```
┌─────────────────────────────────────────┐
│ Workflow Builder                        │
├─────────────────────────────────────────┤
│ Step 1: Metadata                        │
│   - Name (3-80 chars, required)         │
│   - Description (20-240 chars, required)│
│   - Domain(s) (multi-select, required)  │
│                                         │
│ Step 2: Ingestion & Export              │
│   - Upload types (checkboxes)           │
│     ☑ PDF  ☑ DOCX  ☑ TXT  ☑ MD  ☑ CSV │
│   - Ingestion mode (radio)              │
│     ○ Programmatic  ○ Vision            │
│   - Vision prompt (textarea, if vision) │
│   - Export type (select)                │
│     [CSV|JSON|MD|DOCX|PDF]              │
│                                         │
│ Step 3: Prompt Chain                    │
│   [Reuse existing chain editor]         │
│   - Add step title field (required)      │
│                                         │
│ [Cancel] [Save Workflow]                │
└─────────────────────────────────────────┘
```

**Implementation Notes:**
- Can reuse existing `chain-editor-modal` structure
- Add tabs or accordion for 3 steps
- Reuse step rendering logic from current chain editor
- Add domain multi-select (new component)
- Add ingestion/export configuration UI (new)

#### B) Runner UI: Workflow Selection (Redesign)

**Location:** `external/ai_bulk_doc_analysis/templates/bulk_doc_analysis.html`

**Current:** 3-panel layout (Documents, Chains, Runs)  
**New:** Workflow selection → Upload → Run → Download

**Workflow Card Design:**
```html
<div class="workflow-card">
    <div class="workflow-header">
        <h3>Workflow Name</h3>
        <div class="workflow-domains">
            <span class="badge">Risk</span>
            <span class="badge">Compliance</span>
        </div>
    </div>
    <div class="workflow-description">
        Description text (2-line clamp)...
    </div>
    <div class="workflow-steps">
        Steps: Extract obligations → Classify requirements → Generate gaps (+2 more)
    </div>
    <div class="workflow-metadata">
        <span>Inputs: PDF · DOCX</span>
        <span>Ingestion: Programmatic</span>
        <span>Output: CSV</span>
        <span>Steps: 5</span>
        <span>Updated: 3 days ago</span>
    </div>
    <div class="workflow-actions">
        <button>Select & Run</button>
    </div>
</div>
```

**Hidden in Runner UI:**
- Prompt contents
- Vision prompts
- R-selection logic
- Token budgets
- Edit/delete controls

---

## 3. Detailed Implementation Steps

### 3.1 Phase 1: Core Workflow System

#### Step 1.1: Database Migration

**File:** Create `external/ai_bulk_doc_analysis/migrations/0002_workflow_system.py`

```python
"""Migration: Add workflow system tables"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

def upgrade():
    # Create workflows table
    op.create_table(
        'workflows',
        sa.Column('workflow_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('visibility_scope', sa.String(50), nullable=False),
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', JSONB().with_variant(sa.JSON(), 'sqlite'), default={})
    )
    
    # Create workflow_versions table
    op.create_table(
        'workflow_versions',
        sa.Column('workflow_version_id', sa.String(255), primary_key=True),
        sa.Column('workflow_id', sa.String(255), sa.ForeignKey('workflows.workflow_id', ondelete='CASCADE'), nullable=False),
        sa.Column('version_number', sa.Integer(), nullable=False),
        sa.Column('ingestion_profile_id', sa.String(255), nullable=False),
        sa.Column('chain_version_id', sa.String(255), sa.ForeignKey('chain_versions.chain_version_id'), nullable=False),
        sa.Column('export_profile_id', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('workflow_id', 'version_number', name='uq_workflow_version')
    )
    
    # Create workflow_domains table
    op.create_table(
        'workflow_domains',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('workflow_id', sa.String(255), sa.ForeignKey('workflows.workflow_id', ondelete='CASCADE'), nullable=False),
        sa.Column('domain', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('workflow_id', 'domain', name='uq_workflow_domain')
    )
    
    # Create ingestion_profiles table
    op.create_table(
        'ingestion_profiles',
        sa.Column('ingestion_profile_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('accepted_input_types', sa.ARRAY(sa.String()).with_variant(sa.Text(), 'sqlite'), nullable=False),
        sa.Column('mode', sa.String(50), nullable=False),  # 'programmatic' | 'vision'
        sa.Column('vision_prompt', sa.Text(), nullable=True),  # Stored in DB, not file path
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', JSONB().with_variant(sa.JSON(), 'sqlite'), default={})
    )
    
    # Create export_profiles table
    op.create_table(
        'export_profiles',
        sa.Column('export_profile_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('format', sa.String(50), nullable=False),  # 'CSV' | 'JSON' | 'MD' | 'DOCX' | 'PDF'
        sa.Column('config_json', JSONB().with_variant(sa.JSON(), 'sqlite'), default={}),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    
    # Create execution_tasks table (for CSV)
    op.create_table(
        'execution_tasks',
        sa.Column('task_id', sa.String(255), primary_key=True),
        sa.Column('run_id', sa.String(255), sa.ForeignKey('runs.run_id', ondelete='CASCADE'), nullable=False),
        sa.Column('doc_id', sa.String(255), sa.ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False),
        sa.Column('row_index', sa.Integer(), nullable=False),
        sa.Column('row_data', JSONB().with_variant(sa.JSON(), 'sqlite'), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('run_id', 'doc_id', 'row_index', name='uq_execution_task')
    )
    
    # Modify existing tables
    op.add_column('chain_steps', sa.Column('title', sa.String(500), nullable=True))  # Add default later
    op.add_column('runs', sa.Column('workflow_version_id', sa.String(255), nullable=True))
    op.add_column('step_results', sa.Column('task_id', sa.String(255), nullable=True))
    
    # Create indexes
    op.create_index('idx_workflows_created_by', 'workflows', ['created_by'])
    op.create_index('idx_workflow_versions_workflow_id', 'workflow_versions', ['workflow_id'])
    op.create_index('idx_workflow_domains_workflow_id', 'workflow_domains', ['workflow_id'])
    op.create_index('idx_workflow_domains_domain', 'workflow_domains', ['domain'])
    op.create_index('idx_ingestion_profiles_mode', 'ingestion_profiles', ['mode'])
    op.create_index('idx_export_profiles_format', 'export_profiles', ['format'])
    op.create_index('idx_execution_tasks_run_id', 'execution_tasks', ['run_id'])
    op.create_index('idx_execution_tasks_doc_id', 'execution_tasks', ['doc_id'])

def downgrade():
    # Drop indexes
    op.drop_index('idx_execution_tasks_doc_id', 'execution_tasks')
    op.drop_index('idx_execution_tasks_run_id', 'execution_tasks')
    op.drop_index('idx_export_profiles_format', 'export_profiles')
    op.drop_index('idx_ingestion_profiles_mode', 'ingestion_profiles')
    op.drop_index('idx_workflow_domains_domain', 'workflow_domains')
    op.drop_index('idx_workflow_domains_workflow_id', 'workflow_domains')
    op.drop_index('idx_workflow_versions_workflow_id', 'workflow_versions')
    op.drop_index('idx_workflows_created_by', 'workflows')
    
    # Drop columns
    op.drop_column('step_results', 'task_id')
    op.drop_column('runs', 'workflow_version_id')
    op.drop_column('chain_steps', 'title')
    
    # Drop tables
    op.drop_table('execution_tasks')
    op.drop_table('export_profiles')
    op.drop_table('ingestion_profiles')
    op.drop_table('workflow_domains')
    op.drop_table('workflow_versions')
    op.drop_table('workflows')
```

#### Step 1.2: Add SQLAlchemy Models

**File:** `external/ai_bulk_doc_analysis/models.py`

Add new model classes (see IMPLEMENTATION_ANALYSIS.md Section 2.1 for full code).

**Key Points:**
- Use `JSONB().with_variant(JSON(), 'sqlite')` for PostgreSQL/SQLite compatibility
- Add relationships properly
- Use `UniqueConstraint` for composite keys

#### Step 1.3: Create WorkflowService

**File:** `external/ai_bulk_doc_analysis/workflow_service.py` (NEW)

```python
"""
Workflow Service - Manages workflows, versions, and domain associations.
"""
from typing import List, Optional, Dict
from sqlalchemy import and_
from .models import (
    Workflow, WorkflowVersion, WorkflowDomain,
    IngestionProfile, ExportProfile, ChainVersion
)
from .db_service import get_db_session

class WorkflowService:
    def create_workflow(
        self,
        user_id: str,
        name: str,
        description: str,
        domains: List[str],
        ingestion_profile_id: str,
        chain_version_id: str,
        export_profile_id: str
    ) -> Workflow:
        """Create a new workflow with initial version."""
        # Validate description length (20-240 chars)
        if len(description) < 20 or len(description) > 240:
            raise ValueError("Description must be 20-240 characters")
        
        # Validate name length (3-80 chars)
        if len(name) < 3 or len(name) > 80:
            raise ValueError("Name must be 3-80 characters")
        
        # Validate domains
        if not domains or len(domains) == 0:
            raise ValueError("At least one domain is required")
        
        with get_db_session() as db:
            # Create workflow
            workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
            workflow = Workflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                visibility_scope="domain",  # or "super" for super admin
                created_by=user_id
            )
            db.add(workflow)
            db.flush()
            
            # Create domain associations
            for domain in domains:
                wf_domain = WorkflowDomain(
                    workflow_id=workflow_id,
                    domain=domain
                )
                db.add(wf_domain)
            
            # Create initial version
            version_number = 1
            workflow_version_id = f"wfv_{workflow_id}-v{version_number}"
            workflow_version = WorkflowVersion(
                workflow_version_id=workflow_version_id,
                workflow_id=workflow_id,
                version_number=version_number,
                ingestion_profile_id=ingestion_profile_id,
                chain_version_id=chain_version_id,
                export_profile_id=export_profile_id
            )
            db.add(workflow_version)
            db.commit()
            
            return workflow
    
    def update_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        domains: List[str],
        ingestion_profile_id: str,
        chain_version_id: str,
        export_profile_id: str
    ) -> WorkflowVersion:
        """Update workflow - creates new version (immutable)."""
        with get_db_session() as db:
            workflow = db.query(Workflow).filter(Workflow.workflow_id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Update workflow metadata
            workflow.name = name
            workflow.description = description
            workflow.updated_at = datetime.utcnow()
            
            # Update domains (delete old, add new)
            db.query(WorkflowDomain).filter(WorkflowDomain.workflow_id == workflow_id).delete()
            for domain in domains:
                wf_domain = WorkflowDomain(workflow_id=workflow_id, domain=domain)
                db.add(wf_domain)
            
            # Get next version number
            latest_version = (
                db.query(WorkflowVersion)
                .filter(WorkflowVersion.workflow_id == workflow_id)
                .order_by(WorkflowVersion.version_number.desc())
                .first()
            )
            version_number = (latest_version.version_number + 1) if latest_version else 1
            
            # Create new version
            workflow_version_id = f"wfv_{workflow_id}-v{version_number}"
            workflow_version = WorkflowVersion(
                workflow_version_id=workflow_version_id,
                workflow_id=workflow_id,
                version_number=version_number,
                ingestion_profile_id=ingestion_profile_id,
                chain_version_id=chain_version_id,
                export_profile_id=export_profile_id
            )
            db.add(workflow_version)
            db.commit()
            
            return workflow_version
    
    def list_workflows(self, user_id: str, user_domains: List[str], is_super_admin: bool) -> List[Dict]:
        """List workflows visible to user (domain-filtered)."""
        with get_db_session() as db:
            if is_super_admin:
                # Super admin sees all
                workflows = db.query(Workflow).all()
            else:
                # Domain-scoped: workflows that have at least one domain in user's domains
                workflows = (
                    db.query(Workflow)
                    .join(WorkflowDomain)
                    .filter(WorkflowDomain.domain.in_(user_domains))
                    .distinct()
                    .all()
                )
            
            result = []
            for wf in workflows:
                # Get domains
                domains = [wd.domain for wd in wf.domains]
                
                # Get latest version
                latest_version = (
                    db.query(WorkflowVersion)
                    .filter(WorkflowVersion.workflow_id == wf.workflow_id)
                    .order_by(WorkflowVersion.version_number.desc())
                    .first()
                )
                
                result.append({
                    "workflow_id": wf.workflow_id,
                    "name": wf.name,
                    "description": wf.description,
                    "domains": domains,
                    "latest_version_id": latest_version.workflow_version_id if latest_version else None,
                    "created_at": wf.created_at.isoformat() if wf.created_at else None,
                    "updated_at": wf.updated_at.isoformat() if wf.updated_at else None,
                })
            
            return result
```

#### Step 1.4: Create Workflow APIs

**File:** `external/ai_bulk_doc_analysis/blueprint.py`

Add new routes (remove old chain endpoints):

```python
# Remove these:
# @bp.route("/api/bulk-doc-analysis/chains", methods=["GET"])
# @bp.route("/api/bulk-doc-analysis/chains", methods=["POST"])
# @bp.route("/api/bulk-doc-analysis/chains/<chain_id>", methods=["PUT"])

# Add these:
@bp.route("/api/bulk-doc-analysis/workflows", methods=["GET"])
def api_list_workflows():
    """List workflows (domain-filtered)."""
    user_session = _current_user_session()
    if not user_session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user_id = user_session.get("user_id") or "anonymous"
    user_domains = user_session.get("domains", [])
    is_super_admin = auth_manager.is_admin(user_session)  # Assuming super admin check
    
    try:
        from .workflow_service import WorkflowService
        workflow_service = WorkflowService()
        workflows = workflow_service.list_workflows(user_id, user_domains, is_super_admin)
        return jsonify({"workflows": workflows})
    except Exception as e:
        logger.error(f"List workflows error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/api/bulk-doc-analysis/workflows", methods=["POST"])
def api_create_workflow():
    """Create a new workflow."""
    user_session = _current_user_session()
    if not user_session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user_id = user_session.get("user_id") or "anonymous"
    data = request.get_json() or {}
    
    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    domains = data.get("domains", [])
    ingestion_profile_id = data.get("ingestion_profile_id")
    chain_version_id = data.get("chain_version_id")
    export_profile_id = data.get("export_profile_id")
    
    # Validation
    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not description:
        return jsonify({"error": "Description is required"}), 400
    if len(description) < 20 or len(description) > 240:
        return jsonify({"error": "Description must be 20-240 characters"}), 400
    if not domains or len(domains) == 0:
        return jsonify({"error": "At least one domain is required"}), 400
    if not ingestion_profile_id:
        return jsonify({"error": "ingestion_profile_id is required"}), 400
    if not chain_version_id:
        return jsonify({"error": "chain_version_id is required"}), 400
    if not export_profile_id:
        return jsonify({"error": "export_profile_id is required"}), 400
    
    try:
        from .workflow_service import WorkflowService
        workflow_service = WorkflowService()
        workflow = workflow_service.create_workflow(
            user_id=user_id,
            name=name,
            description=description,
            domains=domains,
            ingestion_profile_id=ingestion_profile_id,
            chain_version_id=chain_version_id,
            export_profile_id=export_profile_id
        )
        return jsonify({
            "success": True,
            "workflow": {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
            }
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Create workflow error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/api/bulk-doc-analysis/workflows/<workflow_id>", methods=["PUT"])
def api_update_workflow(workflow_id: str):
    """Update workflow (creates new version)."""
    # Similar to create, but calls update_workflow()
    pass

@bp.route("/api/bulk-doc-analysis/workflows/<workflow_id>", methods=["DELETE"])
def api_delete_workflow(workflow_id: str):
    """Delete workflow."""
    pass
```

---

## 4. UI Implementation Details

### 4.1 Admin Panel: Workflow Builder UI

**File:** `web/templates/admin.html`

**Add new section after "Unstructured Workflows" section:**

```html
<!-- Workflow Builder Section -->
<div class="admin-section">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <h4 style="font-size: 16px; margin: 0; color: #495057; font-weight: 600;">
            🔄 Workflow Builder
        </h4>
        <button class="btn btn-primary btn-sm" onclick="openCreateWorkflowModal()">
            + Create New Workflow
        </button>
    </div>
    <div id="workflows-loading" style="text-align: center; padding: 20px;">
        <div class="spinner-border" role="status"></div>
    </div>
    <div id="workflows-list-container" style="display: none;">
        <ul class="prompt-list" id="workflows-list"></ul>
    </div>
</div>
```

**Workflow Editor Modal (similar to chain editor):**

```html
<div class="prompt-editor-modal" id="workflow-editor-modal">
    <div class="prompt-editor-content" style="max-width: 900px;">
        <div class="prompt-editor-header">
            <h3 id="workflow-editor-title">Create New Workflow</h3>
            <button class="btn btn-secondary" onclick="closeWorkflowEditorModal()">✕ Close</button>
        </div>
        <div class="prompt-editor-body" style="padding: 25px;">
            <!-- Step 1: Metadata -->
            <div id="workflow-step-1" class="workflow-step">
                <h4>Step 1: Metadata</h4>
                <div style="margin-bottom: 20px;">
                    <label>Workflow Name <span class="text-danger">*</span></label>
                    <input type="text" id="workflow-editor-name" class="form-control" 
                           required placeholder="e.g., Policy Review Workflow" 
                           minlength="3" maxlength="80">
                </div>
                <div style="margin-bottom: 20px;">
                    <label>Description <span class="text-danger">*</span></label>
                    <textarea id="workflow-editor-description" class="form-control" rows="3"
                              required placeholder="20-240 characters describing this workflow"
                              minlength="20" maxlength="240"></textarea>
                    <small class="text-muted">20-240 characters required</small>
                </div>
                <div style="margin-bottom: 20px;">
                    <label>Domain(s) <span class="text-danger">*</span></label>
                    <div id="workflow-editor-domains">
                        <!-- Multi-select checkboxes or select2 -->
                        <!-- For now, simple checkboxes -->
                        <div id="domain-checkboxes"></div>
                    </div>
                </div>
            </div>
            
            <!-- Step 2: Ingestion & Export -->
            <div id="workflow-step-2" class="workflow-step" style="display: none;">
                <h4>Step 2: Ingestion & Export</h4>
                <!-- Ingestion profile selection/creation -->
                <!-- Export profile selection/creation -->
            </div>
            
            <!-- Step 3: Prompt Chain -->
            <div id="workflow-step-3" class="workflow-step" style="display: none;">
                <h4>Step 3: Prompt Chain</h4>
                <!-- Reuse existing chain editor -->
            </div>
        </div>
        <div class="prompt-editor-footer">
            <button class="btn btn-secondary" onclick="closeWorkflowEditorModal()">Cancel</button>
            <button class="btn btn-secondary" id="workflow-editor-prev-btn" onclick="prevWorkflowStep()" style="display: none;">Previous</button>
            <button class="btn btn-primary" id="workflow-editor-next-btn" onclick="nextWorkflowStep()">Next</button>
            <button class="btn btn-primary" id="workflow-editor-save-btn" onclick="saveWorkflow()" style="display: none;">Save Workflow</button>
        </div>
    </div>
</div>
```

**JavaScript State Management:**

```javascript
let workflowEditorState = {
    mode: null, // 'create' | 'edit'
    workflowId: null,
    currentStep: 1, // 1, 2, or 3
    name: '',
    description: '',
    domains: [],
    ingestion_profile_id: null,
    chain_version_id: null,
    export_profile_id: null,
    chainSteps: [] // Reuse from chain editor
};

function openCreateWorkflowModal() {
    workflowEditorState = {
        mode: 'create',
        workflowId: null,
        currentStep: 1,
        name: '',
        description: '',
        domains: [],
        ingestion_profile_id: null,
        chain_version_id: null,
        export_profile_id: null,
        chainSteps: []
    };
    renderWorkflowEditor();
    document.getElementById('workflow-editor-modal').classList.add('active');
}

function nextWorkflowStep() {
    if (workflowEditorState.currentStep < 3) {
        workflowEditorState.currentStep++;
        renderWorkflowEditor();
    }
}

function prevWorkflowStep() {
    if (workflowEditorState.currentStep > 1) {
        workflowEditorState.currentStep--;
        renderWorkflowEditor();
    }
}

function renderWorkflowEditor() {
    // Show/hide steps based on currentStep
    // Update button visibility
    // Load domain list for multi-select
    // Reuse chain editor for step 3
}
```

### 4.2 Runner UI: Workflow Cards

**File:** `external/ai_bulk_doc_analysis/templates/bulk_doc_analysis.html`

**Replace chain selection with workflow selection:**

```html
<!-- Panel 2: Workflow Selection (replaces Chain selection) -->
<div class="panel" id="panel-2">
    <h3>Select Workflow</h3>
    
    <!-- Search and filters -->
    <div class="workflow-filters">
        <input type="text" id="workflow-search" placeholder="Search workflows..." class="form-control">
        <select id="domain-filter" class="form-select">
            <option value="">All Domains</option>
            <!-- Populated dynamically -->
        </select>
        <select id="input-type-filter" class="form-select">
            <option value="">All Input Types</option>
            <option value="PDF">PDF</option>
            <option value="DOCX">DOCX</option>
            <!-- etc -->
        </select>
    </div>
    
    <!-- Workflow cards grid -->
    <div class="workflow-cards-grid" id="workflow-cards-container">
        <!-- Populated dynamically -->
    </div>
</div>
```

**Workflow Card Template:**

```javascript
function renderWorkflowCard(workflow) {
    const stepTitles = workflow.step_titles || [];
    const firstThree = stepTitles.slice(0, 3);
    const remaining = stepTitles.length - 3;
    
    return `
        <div class="workflow-card" data-workflow-id="${workflow.workflow_id}">
            <div class="workflow-card-header">
                <h4>${escapeHtml(workflow.name)}</h4>
                <div class="workflow-domains">
                    ${workflow.domains.map(d => `<span class="badge">${escapeHtml(d)}</span>`).join('')}
                </div>
            </div>
            <div class="workflow-card-description">
                ${escapeHtml(workflow.description)}
            </div>
            <div class="workflow-card-steps">
                <strong>Steps:</strong> ${firstThree.join(' → ')}${remaining > 0 ? ` <span class="text-muted">(+${remaining} more)</span>` : ''}
            </div>
            <div class="workflow-card-metadata">
                <span>Inputs: ${workflow.accepted_input_types.join(' · ')}</span>
                <span>Ingestion: ${workflow.ingestion_mode === 'vision' ? 'Vision (LLM)' : 'Programmatic'}</span>
                <span>Output: ${workflow.export_type}</span>
                <span>Steps: ${workflow.step_count}</span>
                <span>Updated: ${formatRelativeTime(workflow.updated_at)}</span>
            </div>
            <div class="workflow-card-actions">
                <button class="btn btn-primary" onclick="selectWorkflow('${workflow.workflow_id}')">
                    Select & Run
                </button>
            </div>
        </div>
    `;
}
```

---

## 5. Key Implementation Notes

### 5.1 Step Title Field

**Current:** Steps don't have titles  
**New:** Each step must have a `title` field (required)

**Changes needed:**
1. Database: Add `title` column to `chain_steps`
2. Models: Add `title` field to `ChainStep` model
3. Validation: Require title in chain validation
4. UI: Add title input field in step editor
5. API: Include title in step creation/update

**UI Change in `renderChainStep()`:**
```javascript
// Add before "Required Inputs" section:
<div style="margin-bottom: 15px;">
    <label style="font-size: 0.9em; font-weight: 600; display: block; margin-bottom: 5px;">
        Step Title <span class="text-danger">*</span>
    </label>
    <input type="text" class="form-control" 
           data-step-index="${step.index}" data-field="title" 
           placeholder="e.g., Extract policy sections"
           value="${escapeHtml(step.title || '')}" required>
</div>
```

### 5.2 Domain System

**New:** Build from scratch

**Implementation:**
1. Extend `users.json`:
   ```json
   {
     "user_id": "user1",
     "domains": ["Risk", "Compliance"]  // NEW
   }
   ```

2. Update `AuthManager`:
   ```python
   def get_user_domains(self, user_id: str) -> List[str]:
       user = self.users.get(user_id)
       return user.get("domains", []) if user else []
   ```

3. Domain access control:
   ```python
   def can_view_workflow(user_session, workflow):
       if is_super_admin(user_session):
           return True
       user_domains = user_session.get("domains", [])
       workflow_domains = get_workflow_domains(workflow.workflow_id)
       return bool(set(user_domains) & set(workflow_domains))
   ```

### 5.3 Vision Prompt Storage

**Decision:** Store in database (TEXT column), not file path

**Implementation:**
- `ingestion_profiles.vision_prompt` = TEXT column
- UI: Textarea (not file upload)
- Validation: Required if mode='vision'

### 5.4 CSV Handling

**Special:** Row = task, not file = task

**Implementation:**
- On CSV upload: Parse CSV, create `execution_tasks` per row
- R0 for task: Serialized row data (JSON or MD)
- Execution: Per-task, not per-doc
- Export: Compile from task outputs

---

## 6. Testing Checklist

### 6.1 Unit Tests

- [ ] Workflow CRUD operations
- [ ] Domain filtering logic
- [ ] Workflow versioning (immutable)
- [ ] Step title validation
- [ ] Ingestion profile creation
- [ ] Export profile creation

### 6.2 Integration Tests

- [ ] Create workflow → Create version → List workflows
- [ ] Domain access control (user sees only their domains)
- [ ] Workflow update creates new version
- [ ] CSV task creation and execution
- [ ] Vision ingestion (PDF → images → Claude)

### 6.3 UI Tests

- [ ] Admin: Create workflow (all 3 steps)
- [ ] Admin: Edit workflow (creates version)
- [ ] Admin: Domain multi-select works
- [ ] Runner: Workflow cards display correctly
- [ ] Runner: Search and filtering works
- [ ] Runner: Step titles shown (first 3, then "+N more")

---

## 7. Migration Path

### 7.1 Remove Old Endpoints

**Files to modify:**
- `external/ai_bulk_doc_analysis/blueprint.py` - Remove chain endpoints
- `web/templates/admin.html` - Remove or hide "Unstructured Workflows" section (or convert to workflow builder)

### 7.2 Data Migration

**No migration needed** - Fresh start with workflow system.

**Optional:** If you want to preserve existing chains:
- Create workflows from existing chains
- Create default ingestion_profile (PDF, programmatic)
- Create default export_profile (MD)
- Link via workflow_version

---

## 8. Dependencies

Add to `requirements.txt`:
```txt
python-docx>=1.1.0
PyMuPDF>=1.23.0
pandas>=2.0.0
Pillow>=10.0.0
pdf2image>=1.16.0
reportlab>=4.0.0
```

---

## 9. File Structure Summary

### New Files
```
external/ai_bulk_doc_analysis/
├── workflow_service.py          # NEW
├── ingestion_service.py         # NEW
├── export_service.py            # NEW
├── domain_service.py            # NEW (optional)
├── migrations/
│   └── 0002_workflow_system.py  # NEW
└── workers/
    └── export_worker.py         # NEW
```

### Modified Files
```
external/ai_bulk_doc_analysis/
├── models.py                    # Add new models
├── db_schema.sql                # Add new tables
├── db_service.py                # Add workflow methods
├── services.py                  # Enhance for multi-file
├── blueprint.py                 # Remove old endpoints, add new
└── workers/
    ├── conversion_worker.py     # Multi-format + vision
    └── execution_worker.py      # CSV task support

web/templates/
└── admin.html                   # Add workflow builder section

external/ai_bulk_doc_analysis/templates/
└── bulk_doc_analysis.html       # Redesign Runner UI

config/
└── users.json                   # Add domains field
```

---

## 10. Next Steps

1. **Review this document** thoroughly
2. **Start with Phase 1** (Core Workflow System)
3. **Create database migration** first
4. **Add models** to `models.py`
5. **Create WorkflowService**
6. **Add APIs** to `blueprint.py`
7. **Test APIs** with Postman/curl
8. **Build UI** incrementally

---

**End of Build Details**

