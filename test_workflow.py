#!/usr/bin/env python3
"""
Script to create and test a simple workflow:
- PDF input
- One-step chain (extract obligations)
- Markdown output
"""

import os
import sys
import json
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER = "admin"  # Assuming admin user exists

# Sample files
SAMPLE_DIR = project_root / "Agents at Scale" / "sample_workflow_test_pack 2"
PDF_FILE = SAMPLE_DIR / "input_policy_with_table.pdf"
PROMPT_FILE = SAMPLE_DIR / "prompts" / "step_01_extract_obligations.md"

def get_auth_token():
    """Get auth token - you may need to adjust this based on your auth setup"""
    # For now, we'll assume session-based auth
    # You may need to login first or use a token
    return None

def make_request(method, endpoint, data=None, files=None):
    """Make HTTP request to API"""
    url = f"{BASE_URL}{endpoint}"
    headers = {}
    
    # Add auth if needed
    token = get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    if method == "GET":
        response = requests.get(url, headers=headers)
    elif method == "POST":
        if files:
            response = requests.post(url, headers=headers, data=data, files=files)
        else:
            headers["Content-Type"] = "application/json"
            response = requests.post(url, headers=headers, json=data)
    elif method == "PUT":
        headers["Content-Type"] = "application/json"
        response = requests.put(url, headers=headers, json=data)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    response.raise_for_status()
    return response.json()

def create_ingestion_profile():
    """Create ingestion profile for PDF (programmatic)"""
    print("Creating ingestion profile...")
    
    # Use direct DB access instead of API
    from external.ai_bulk_doc_analysis.ingestion_service import IngestionService
    from external.ai_bulk_doc_analysis.db_service import init_db
    
    init_db()
    ingestion_service = IngestionService()
    
    # Create profile and get ID within the service method's session
    from external.ai_bulk_doc_analysis.db_service import get_db_session
    from external.ai_bulk_doc_analysis.models import IngestionProfile
    import uuid
    
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
        profile_id = profile.ingestion_profile_id  # Access while in session
    
    print(f"✓ Created ingestion profile: {profile_id}")
    return profile_id

def create_export_profile():
    """Create export profile for Markdown"""
    print("Creating export profile...")
    
    # Use direct DB access instead of API
    from external.ai_bulk_doc_analysis.export_service import ExportService
    from external.ai_bulk_doc_analysis.db_service import init_db
    
    init_db()
    
    # Create profile and get ID within session
    from external.ai_bulk_doc_analysis.db_service import get_db_session
    from external.ai_bulk_doc_analysis.models import ExportProfile
    import uuid
    
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
        profile_id = profile.export_profile_id  # Access while in session
    
    print(f"✓ Created export profile: {profile_id}")
    return profile_id

def create_chain():
    """Create a chain with one step (extract obligations)"""
    print("Creating chain...")
    
    # Read prompt
    prompt_text = PROMPT_FILE.read_text()
    
    # Create chain via direct DB access (since there's no API endpoint)
    # We'll use the db_service directly
    from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db, get_db_session
    from external.ai_bulk_doc_analysis.models import Chain, ChainStep, ChainVersion
    
    init_db()
    svc = BulkDocDBService()
    
    # Create chain
    chain = svc.create_chain(
        user_id=TEST_USER,
        name="Simple Obligation Extractor",
        description="Extract obligations from documents",
        steps=[
            {
                "index": 1,
                "title": "Extract Obligations",
                "prompt": prompt_text,
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
    
    # Get the latest version - chain object should have version info
    # The create_chain returns a Chain dataclass, we need to get the version_id
    # Let's query for it
    with get_db_session() as db:
        latest_version = db.query(ChainVersion).filter(
            ChainVersion.chain_id == chain.chain_id
        ).order_by(ChainVersion.version_number.desc()).first()
        
        if not latest_version:
            raise ValueError(f"No version found for chain {chain.chain_id}")
        
        chain_version_id = latest_version.chain_version_id
        print(f"✓ Created chain: {chain.chain_id}, version: {chain_version_id}")
        return chain_version_id

def create_workflow(ingestion_profile_id, export_profile_id, chain_version_id):
    """Create workflow"""
    print("Creating workflow...")
    
    # Use direct DB access instead of API
    from external.ai_bulk_doc_analysis.workflow_service import WorkflowService
    from external.ai_bulk_doc_analysis.db_service import init_db
    
    init_db()
    workflow_service = WorkflowService()
    
    # Create workflow and get ID from within the service
    from external.ai_bulk_doc_analysis.db_service import get_db_session
    from external.ai_bulk_doc_analysis.models import Workflow, WorkflowVersion, WorkflowDomain, ChainVersion
    import uuid
    
    with get_db_session() as db:
        # Verify chain_version exists
        chain_version = db.query(ChainVersion).filter(
            ChainVersion.chain_version_id == chain_version_id
        ).first()
        if not chain_version:
            raise ValueError(f"Chain version {chain_version_id} not found")
        
        # Create workflow
        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        workflow = Workflow(
            workflow_id=workflow_id,
            name="PDF Obligation Extractor",
            description="Extract obligations from PDF documents and output as Markdown",
            visibility_scope="domain",
            created_by=TEST_USER
        )
        db.add(workflow)
        db.flush()
        
        # Create domain associations
        for domain in ["Compliance", "Risk"]:
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
        
        # Get workflow_id while still in session
        workflow_id_result = workflow_id
    
    print(f"✓ Created workflow: {workflow_id_result}")
    return {"workflow_id": workflow_id_result}

def upload_document(workflow_version_id):
    """Upload PDF document"""
    print(f"Uploading PDF: {PDF_FILE.name}...")
    
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_FILE}")
    
    # Use direct DB access
    from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db
    from external.ai_bulk_doc_analysis.workflow_service import WorkflowService
    
    init_db()
    svc = BulkDocDBService()
    workflow_service = WorkflowService()
    
    # Get workflow version to get ingestion profile
    from external.ai_bulk_doc_analysis.db_service import get_db_session
    from external.ai_bulk_doc_analysis.models import WorkflowVersion
    
    with get_db_session() as db:
        wf_version = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_version_id == workflow_version_id
        ).first()
        if not wf_version:
            raise ValueError(f"Workflow version {workflow_version_id} not found")
        
        ingestion_profile_id = wf_version.ingestion_profile_id
    
    # Create session
    session_id = svc.create_session(TEST_USER)
    
    # Create a file-like object for upload
    from io import BytesIO
    from werkzeug.datastructures import FileStorage
    
    with open(PDF_FILE, 'rb') as f:
        file_content = f.read()
    
    # Create FileStorage object
    file_storage = FileStorage(
        stream=BytesIO(file_content),
        filename=PDF_FILE.name,
        content_type='application/pdf'
    )
    
    # Upload document using create_documents
    docs = svc.create_documents(
        session_id=session_id,
        files=[file_storage],
        user_id=TEST_USER
    )
    
    if not docs:
        raise ValueError("Failed to upload document")
    
    doc = docs[0]
    doc_id = doc.doc_id
    
    # Convert document (programmatic ingestion)
    from external.ai_bulk_doc_analysis.ingestion_service import IngestionService
    ingestion_service = IngestionService()
    
    # Get ingestion profile
    ingestion_profile = ingestion_service.get_ingestion_profile(ingestion_profile_id)
    if not ingestion_profile:
        raise ValueError(f"Ingestion profile {ingestion_profile_id} not found")
    
    # Get document file path
    doc_dir = svc.storage_base / "docs" / doc_id
    file_path = None
    for ext in ['.pdf', '.docx', '.txt', '.md', '.csv']:
        for f in doc_dir.glob(f"*{ext}"):
            file_path = f
            break
        if file_path:
            break
    
    if not file_path:
        raise ValueError(f"File not found for document {doc_id}")
    
    # Ingest file
    print(f"Converting document...")
    r0_content, metadata = ingestion_service.ingest_file(
        file_path=file_path,
        ingestion_profile=ingestion_profile
    )
    
    # Save R0 content and update document status
    r0_path = doc_dir / "R0.md"
    r0_path.write_text(r0_content, encoding='utf-8')
    
    # Update document status to CONVERTED
    svc.update_document_status(
        doc_id=doc_id,
        status="CONVERTED",
        converted_md_path=str(r0_path.relative_to(svc.storage_base))
    )
    
    print(f"✓ Uploaded and converted document: {doc_id}")
    return doc_id, session_id

def create_run(session_id, workflow_version_id):
    """Create a run"""
    print("Creating run...")
    
    # Use direct DB access
    from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db
    from external.ai_bulk_doc_analysis.workflow_service import WorkflowService
    
    init_db()
    svc = BulkDocDBService()
    workflow_service = WorkflowService()
    
    # Get workflow version to get chain_version_id
    from external.ai_bulk_doc_analysis.db_service import get_db_session
    from external.ai_bulk_doc_analysis.models import WorkflowVersion
    
    with get_db_session() as db:
        wf_version = db.query(WorkflowVersion).filter(
            WorkflowVersion.workflow_version_id == workflow_version_id
        ).first()
        if not wf_version:
            raise ValueError(f"Workflow version {workflow_version_id} not found")
        
        chain_version_id = wf_version.chain_version_id
    
    # Create run
    run = svc.create_run(
        session_id=session_id,
        chain_version_id=chain_version_id,
        workflow_version_id=workflow_version_id
    )
    
    run_id = run.run_id
    print(f"✓ Created run: {run_id}")
    return run_id

def check_run_progress(run_id):
    """Check run progress"""
    print(f"Checking run progress: {run_id}...")
    
    # Use direct DB access
    from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db
    
    init_db()
    svc = BulkDocDBService()
    
    progress = svc.get_run_progress(run_id)
    if not progress:
        raise ValueError(f"Run {run_id} not found")
    
    print(f"Run status: {progress.get('status')}")
    print(f"Progress: {progress.get('progress', {})}")
    return progress

def main():
    """Main test flow"""
    print("=" * 60)
    print("Testing Simple Workflow: PDF -> Extract Obligations -> MD")
    print("=" * 60)
    
    try:
        # Step 1: Create ingestion profile
        ingestion_profile_id = create_ingestion_profile()
        
        # Step 2: Create export profile
        export_profile_id = create_export_profile()
        
        # Step 3: Create chain
        chain_version_id = create_chain()
        
        # Step 4: Create workflow
        workflow_result = create_workflow(ingestion_profile_id, export_profile_id, chain_version_id)
        workflow_id = workflow_result['workflow_id']
        
        # Step 5: Get workflow version (we need the version_id, not workflow_id)
        # The workflow creation should return the version_id
        from external.ai_bulk_doc_analysis.workflow_service import WorkflowService
        workflow_service = WorkflowService()
        workflow = workflow_service.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Get latest version - workflow is a dict with workflow info
        # We need to query for the latest version
        from external.ai_bulk_doc_analysis.db_service import get_db_session
        from external.ai_bulk_doc_analysis.models import WorkflowVersion
        
        with get_db_session() as db:
            latest_version = db.query(WorkflowVersion).filter(
                WorkflowVersion.workflow_id == workflow_id
            ).order_by(WorkflowVersion.version_number.desc()).first()
            
            if not latest_version:
                raise ValueError(f"No version found for workflow {workflow_id}")
            
            workflow_version_id = latest_version.workflow_version_id
            print(f"Using workflow version: {workflow_version_id}")
        
        # Step 6: Upload document
        doc_id, session_id = upload_document(workflow_version_id)
        
        # Step 7: Create run
        run_id = create_run(session_id, workflow_version_id)
        
        # Step 8: Check progress (wait a bit first)
        import time
        print("\nWaiting 5 seconds for processing to start...")
        time.sleep(5)
        
        progress = check_run_progress(run_id)
        
        # Step 9: Wait for completion (poll)
        max_wait = 120  # 2 minutes
        wait_time = 0
        while wait_time < max_wait:
            progress = check_run_progress(run_id)
            status = progress.get('status')
            
            if status == 'COMPLETE':
                print("\n✓ Run completed successfully!")
                break
            elif status == 'ERROR':
                print("\n✗ Run failed!")
                print(f"Error: {progress.get('error_message', 'Unknown error')}")
                break
            
            print(f"Waiting... ({wait_time}s/{max_wait}s)")
            time.sleep(5)
            wait_time += 5
        
        if wait_time >= max_wait:
            print("\n⚠ Run did not complete within timeout")
        
        # Step 10: Download results
        if progress.get('status') == 'COMPLETE':
            print("\nDownloading results...")
            # Get document output using service
            from external.ai_bulk_doc_analysis.db_service import BulkDocDBService, init_db
            
            init_db()
            svc = BulkDocDBService()
            
            # Get final output path
            chain = svc.get_chain(progress.get('chain_version_id'))
            if chain:
                output_path = svc.get_final_output_path(run_id, doc_id, chain)
                if output_path and output_path.exists():
                    output_file = project_root / "test_output.md"
                    import shutil
                    shutil.copy(output_path, output_file)
                    print(f"✓ Results saved to: {output_file}")
                    print(f"\nFirst 500 chars of output:")
                    print(output_file.read_text()[:500])
                else:
                    print(f"✗ Output file not found at: {output_path}")
            else:
                print(f"✗ Chain not found")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

