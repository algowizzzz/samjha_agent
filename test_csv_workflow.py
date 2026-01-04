#!/usr/bin/env python3
"""Test CSV workflow: Task Analysis & Recommendations"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
session = requests.Session()

def login():
    """Login and get session cookie"""
    login_data = {"user_id": "admin", "password": "admin123"}
    response = session.post(
        f"{BASE_URL}/login",
        data=login_data,
        allow_redirects=True
    )
    if response.status_code == 200 and "/login" not in response.url:
        print("✓ Logged in")
    else:
        raise Exception(f"Login failed: {response.status_code}")

def create_ingestion_profile():
    """Create CSV ingestion profile"""
    response = session.post(
        f"{BASE_URL}/api/bulk-doc-analysis/ingestion-profiles",
        json={
            "name": "CSV Test Profile",
            "accepted_input_types": ["CSV"],
            "mode": "programmatic"
        }
    )
    response.raise_for_status()
    result = response.json()
    profile = result.get("profile", result)
    profile_id = profile.get("ingestion_profile_id")
    print(f"✓ Created ingestion profile: {profile_id}")
    return profile_id

def create_export_profile():
    """Create Markdown export profile"""
    response = session.post(
        f"{BASE_URL}/api/bulk-doc-analysis/export-profiles",
        json={
            "name": "MD Export Profile",
            "format": "MD",
            "config_json": {}
        }
    )
    response.raise_for_status()
    result = response.json()
    profile = result.get("profile", result)
    profile_id = profile.get("export_profile_id")
    print(f"✓ Created export profile: {profile_id}")
    return profile_id

def create_chain():
    """Create 2-step chain for task analysis"""
    response = session.post(
        f"{BASE_URL}/api/bulk-doc-analysis/chains",
        json={
            "name": "Task Analysis Chain",
            "description": "2-step chain for CSV task processing",
            "steps": [
                {
                    "index": 1,
                    "title": "Analyze Task",
                    "prompt": """You are analyzing a support ticket. Review the CSV row data provided in the R0 Input section below.

Extract and analyze the following information from the ticket:
- Ticket ID
- Customer name
- Issue description
- Priority level

Based on this information, provide:
1. **Category**: What type of task is this? (e.g., Document Processing, Data Extraction, Format Conversion, Compliance Review)
2. **Key Requirements**: List 2-3 main requirements or goals from the issue description
3. **Complexity Assessment**: Rate as Low, Medium, or High and explain why
4. **Dependencies**: What other information or resources might be needed to complete this task?

Format your response in clear markdown with headers and bullet points.""",
                    "description": "",
                    "required_inputs": ["R0"],
                    "model_config": {
                        "model": "claude-haiku-4-5-20251001",
                        "temperature": 0.3,
                        "max_tokens": 1000
                    }
                },
                {
                    "index": 2,
                    "title": "Generate Recommendations",
                    "prompt": """Based on the task analysis provided in R1, generate actionable recommendations for handling this ticket.

Review the original ticket data from R0 Input and the analysis from R1 Input.

**Provide:**
1. **Recommended Approach**: Step-by-step approach to resolve the issue (numbered list)
2. **Estimated Effort**: Time/resources needed (Low/Medium/High) with justification
3. **Risk Factors**: Potential challenges or blockers that might arise
4. **Next Steps**: Specific, actionable next steps to take (numbered list)

Format as a structured markdown document with clear sections and headers.""",
                    "description": "",
                    "required_inputs": ["R0", "R1"],
                    "model_config": {
                        "model": "claude-haiku-4-5-20251001",
                        "temperature": 0.3,
                        "max_tokens": 1200
                    }
                }
            ]
        }
    )
    response.raise_for_status()
    result = response.json()
    chain = result.get("chain", result)
    chain_id = chain.get("chain_id")
    chain_version_id = chain.get("chain_version_id")
    print(f"✓ Created chain: {chain_id}")
    return chain_id, chain_version_id

def create_workflow(ingestion_profile_id, chain_version_id, export_profile_id):
    """Create workflow"""
    response = session.post(
        f"{BASE_URL}/api/bulk-doc-analysis/workflows",
        json={
            "name": "Task Analysis & Recommendations",
            "description": "Process task tickets from CSV, analyze issues, and generate recommendations",
            "domains": ["Operations"],
            "ingestion_profile_id": ingestion_profile_id,
            "chain_version_id": chain_version_id,
            "export_profile_id": export_profile_id
        }
    )
    response.raise_for_status()
    result = response.json()
    workflow_data = result.get("workflow", result)
    workflow_id = workflow_data.get("workflow_id")
    workflow_version_id = workflow_data.get("workflow_version_id")
    print(f"✓ Created workflow: {workflow_id}")
    return workflow_id, workflow_version_id

def upload_csv(workflow_version_id):
    """Upload CSV file"""
    csv_path = Path("test_files/sample_tasks.csv")
    with open(csv_path, "rb") as f:
        files = {"files": (csv_path.name, f, "text/csv")}
        data = {"workflow_version_id": workflow_version_id}
        response = session.post(
            f"{BASE_URL}/api/bulk-doc-analysis/documents/upload",
            files=files,
            data=data
        )
    response.raise_for_status()
    result = response.json()
    # API may return a list of documents
    docs = result.get("documents", [])
    if docs:
        doc_id = docs[0].get("doc_id")
    else:
        doc = result.get("document", result)
        doc_id = doc.get("doc_id") if isinstance(doc, dict) else None
    print(f"✓ Uploaded CSV: {doc_id}")
    return doc_id

def wait_for_conversion(doc_id, timeout=60):
    """Wait for document conversion"""
    print(f"Waiting for conversion of {doc_id}...")
    start = time.time()
    while time.time() - start < timeout:
        # List all documents and find ours
        response = session.get(f"{BASE_URL}/api/bulk-doc-analysis/documents")
        response.raise_for_status()
        result = response.json()
        docs = result.get("documents", [])
        
        doc = next((d for d in docs if d.get("doc_id") == doc_id), None)
        if not doc:
            print(f"  Document {doc_id} not found in list")
            time.sleep(2)
            continue
            
        status = doc.get("status")
        print(f"  Status: {status}")
        if status == "CONVERTED":
            print("✓ Conversion complete")
            return True
        elif status == "ERROR":
            print(f"✗ Conversion failed: {doc.get('error_message', 'Unknown error')}")
            return False
        time.sleep(2)
    print("✗ Conversion timeout")
    return False

def create_run(workflow_version_id, doc_id):
    """Create run"""
    response = session.post(
        f"{BASE_URL}/api/bulk-doc-analysis/runs",
        json={
            "workflow_version_id": workflow_version_id,
            "doc_ids": [doc_id]
        }
    )
    response.raise_for_status()
    result = response.json()
    run_data = result.get("run", result)
    run_id = run_data.get("run_id")
    print(f"✓ Created run: {run_id}")
    return run_id

def wait_for_run_completion(run_id, timeout=300):
    """Wait for run to complete"""
    print(f"Waiting for run {run_id} to complete...")
    start = time.time()
    while time.time() - start < timeout:
        response = session.get(f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/progress")
        if response.status_code == 404:
            print("  Run not found yet, waiting...")
            time.sleep(2)
            continue
        response.raise_for_status()
        result = response.json()
        status = result.get("status")
        print(f"  Status: {status}")
        
        if status == "COMPLETE":
            print("✓ Run complete")
            return True
        elif status == "ERROR":
            print(f"✗ Run failed")
            return False
        
        # Show progress
        rows = result.get("rows", [])
        if rows:
            statuses = {}
            for row in rows:
                s = row.get("status")
                statuses[s] = statuses.get(s, 0) + 1
            print(f"  Progress: {statuses}")
        
        time.sleep(5)
    print("✗ Run timeout")
    return False

def download_outputs(run_id):
    """Download all outputs"""
    response = session.get(
        f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/download-all",
        stream=True
    )
    response.raise_for_status()
    
    output_dir = Path("test_csv_outputs")
    output_dir.mkdir(exist_ok=True)
    
    zip_path = output_dir / f"run_{run_id[:8]}.zip"
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✓ Downloaded outputs to {zip_path}")
    return zip_path

def main():
    print("=" * 60)
    print("Testing CSV Task Processing Workflow")
    print("=" * 60)
    
    try:
        login()
        
        # Create profiles and workflow
        ingestion_profile_id = create_ingestion_profile()
        export_profile_id = create_export_profile()
        chain_id, chain_version_id = create_chain()
        workflow_id, workflow_version_id = create_workflow(
            ingestion_profile_id, chain_version_id, export_profile_id
        )
        
        # Upload and process CSV
        doc_id = upload_csv(workflow_version_id)
        if not wait_for_conversion(doc_id):
            return
        
        # Create run
        run_id = create_run(workflow_version_id, doc_id)
        if not wait_for_run_completion(run_id):
            return
        
        # Download outputs
        zip_path = download_outputs(run_id)
        
        print("\n" + "=" * 60)
        print("✓ Test completed successfully!")
        print(f"Outputs downloaded to: {zip_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

