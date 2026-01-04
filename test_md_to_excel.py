#!/usr/bin/env python3
"""
Test MD to Excel conversion with three test markdown files.
Tests: Upload 3 MD files → Create workflow with XLSX export → Run → Download Excel
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"

def get_session_cookies():
    """Get session cookies by logging in"""
    login_url = f"{BASE_URL}/login"
    login_data = {"user_id": "admin", "password": "admin123"}
    
    try:
        session = requests.Session()
        response = session.post(login_url, data=login_data, allow_redirects=True)
        if response.status_code == 200 and "/login" not in response.url:
            print("✓ Login successful")
            return session
        else:
            print(f"⚠️  Login failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  Login error: {e}")
        return None

def create_ingestion_profile(session):
    """Create ingestion profile for MD files"""
    print("\n" + "="*60)
    print("Creating Ingestion Profile for MD")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/ingestion-profiles"
    data = {
        "name": "MD Test Profile",
        "accepted_input_types": ["MD"],
        "mode": "programmatic"
    }
    
    response = session.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        # API returns: {"success": True, "profile": {...}}
        profile = result.get('profile', result)
        profile_id = profile.get('ingestion_profile_id')
        print(f"✓ Created ingestion profile: {profile_id}")
        return profile_id
    else:
        print(f"❌ Failed to create ingestion profile: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def create_export_profile(session):
    """Create export profile for XLSX"""
    print("\n" + "="*60)
    print("Creating Export Profile for XLSX (Excel)")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/export-profiles"
    data = {
        "name": "Excel Export Profile",
        "format": "XLSX",
        "config_json": {}
    }
    
    response = session.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        # API returns: {"success": True, "profile": {...}}
        profile = result.get('profile', result)
        profile_id = profile.get('export_profile_id')
        print(f"✓ Created export profile: {profile_id}")
        return profile_id
    else:
        print(f"❌ Failed to create export profile: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def create_chain(session):
    """Create a simple 1-step chain"""
    print("\n" + "="*60)
    print("Creating Chain")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains"
    data = {
        "name": "MD to Excel Test Chain",
        "description": "Test chain for MD to Excel conversion",
        "steps": [
            {
                "index": 1,
                "title": "Extract Content",
                "prompt": "Extract all content from the markdown document. Preserve structure, tables, and formatting.",
                "required_inputs": ["R0"],
                "model_config": {
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 4096,
                    "temperature": 0.2
                }
            }
        ]
    }
    
    response = session.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        # API returns: {"success": True, "chain": {...}}
        chain = result.get('chain', result)
        chain_id = chain.get('chain_id')
        chain_version_id = chain.get('chain_version_id')
        print(f"✓ Created chain: {chain_id} (version: {chain_version_id})")
        return chain_id, chain_version_id
    else:
        print(f"❌ Failed to create chain: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def create_workflow(session, ingestion_profile_id, chain_version_id, export_profile_id):
    """Create workflow"""
    print("\n" + "="*60)
    print("Creating Workflow")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/workflows"
    data = {
        "name": "MD to Excel Test Workflow",
        "description": "Test workflow for converting MD files to Excel format with multiple sheets",
        "domains": ["testing"],
        "ingestion_profile_id": ingestion_profile_id,
        "chain_version_id": chain_version_id,
        "export_profile_id": export_profile_id
    }
    
    response = session.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        # API returns: {"success": True, "workflow": {...}}
        workflow = result.get('workflow', result)
        workflow_id = workflow.get('workflow_id')
        workflow_version_id = workflow.get('workflow_version_id') or workflow.get('version_id')
        print(f"✓ Created workflow: {workflow_id} (version: {workflow_version_id})")
        if not workflow_version_id:
            print(f"   Warning: workflow_version_id not found in response: {workflow}")
        return workflow_id, workflow_version_id
    else:
        print(f"❌ Failed to create workflow: {response.status_code}")
        print(f"   Response: {response.text}")
        return None, None

def get_current_session(session):
    """Get current session ID"""
    url = f"{BASE_URL}/api/bulk-doc-analysis/sessions"
    response = session.get(url)
    if response.status_code == 200:
        sessions = response.json().get('sessions', [])
        if sessions:
            return sessions[0].get('session_id')
    return None

def upload_md_file(session, session_id, file_path, workflow_version_id):
    """Upload an MD file"""
    url = f"{BASE_URL}/api/bulk-doc-analysis/test-files/upload"
    
    data = {
        "file_path": str(file_path),
        "workflow_version_id": workflow_version_id
    }
    
    response = session.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Uploaded {file_path.name}: {result.get('document', {}).get('doc_id')}")
        return result.get('document', {}).get('doc_id')
    else:
        print(f"❌ Failed to upload {file_path.name}: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def wait_for_conversion(session, session_id, timeout=60):
    """Wait for documents to be converted"""
    print("\n" + "="*60)
    print("Waiting for document conversion...")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/sessions/{session_id}/documents"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = session.get(url)
        if response.status_code == 200:
            docs = response.json().get('documents', [])
            all_converted = all(d.get('status') == 'CONVERTED' for d in docs)
            if all_converted:
                print("✓ All documents converted")
                return True
            else:
                statuses = [d.get('status') for d in docs]
                print(f"   Statuses: {statuses}")
        time.sleep(2)
    
    print("⚠️  Timeout waiting for conversion")
    return False

def create_run(session, session_id, workflow_version_id):
    """Create a run"""
    print("\n" + "="*60)
    print("Creating Run")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs"
    data = {
        "workflow_version_id": workflow_version_id
    }
    
    response = session.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        # API returns: {"run": {...}} or direct run object
        run = result.get('run', result)
        run_id = run.get('run_id')
        print(f"✓ Created run: {run_id}")
        return run_id
    else:
        print(f"❌ Failed to create run: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def wait_for_run_completion(session, run_id, timeout=300):
    """Wait for run to complete"""
    print("\n" + "="*60)
    print("Waiting for run completion...")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/progress"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = session.get(url)
        if response.status_code == 200:
            progress = response.json()
            status = progress.get('status')
            print(f"   Run status: {status}")
            
            if status == 'COMPLETE':
                print("✓ Run completed")
                return True
            elif status == 'ERROR':
                print("❌ Run failed")
                return False
            
            # Show progress
            if 'rows' in progress:
                rows = progress.get('rows', [])
                success = sum(1 for r in rows if r.get('status') == 'SUCCESS')
                total = len(rows)
                print(f"   Progress: {success}/{total} documents completed")
        time.sleep(5)
    
    print("⚠️  Timeout waiting for run completion")
    return False

def download_output(session, run_id, doc_id, output_path):
    """Download output file"""
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/download/{doc_id}"
    
    response = session.get(url)
    if response.status_code == 200:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Downloaded output to: {output_path}")
        return True
    else:
        print(f"❌ Failed to download output: {response.status_code}")
        print(f"   Response: {response.text}")
        return False

def main():
    print("="*60)
    print("MD to Excel Conversion Test")
    print("="*60)
    
    # Login
    session = get_session_cookies()
    if not session:
        print("❌ Cannot proceed without login")
        return
    
    # Get current session
    session_id = get_current_session(session)
    if not session_id:
        print("❌ Cannot get session ID")
        return
    
    print(f"✓ Using session: {session_id}")
    
    # Create profiles and chain
    ingestion_profile_id = create_ingestion_profile(session)
    if not ingestion_profile_id:
        return
    
    export_profile_id = create_export_profile(session)
    if not export_profile_id:
        return
    
    chain_id, chain_version_id = create_chain(session)
    if not chain_id:
        return
    
    # Create workflow
    workflow_id, workflow_version_id = create_workflow(session, ingestion_profile_id, chain_version_id, export_profile_id)
    if not workflow_version_id:
        return
    
    # Upload test files
    print("\n" + "="*60)
    print("Uploading Test MD Files")
    print("="*60)
    
    test_files = [
        Path("md_to_excel_testcase_1.md"),
        Path("md_to_excel_testcase_2.md"),
        Path("md_to_excel_testcase_3.md")
    ]
    
    doc_ids = []
    for test_file in test_files:
        if not test_file.exists():
            print(f"⚠️  File not found: {test_file}")
            continue
        doc_id = upload_md_file(session, session_id, test_file, workflow_version_id)
        if doc_id:
            doc_ids.append(doc_id)
        time.sleep(1)  # Small delay between uploads
    
    if not doc_ids:
        print("❌ No files uploaded")
        return
    
    print(f"✓ Uploaded {len(doc_ids)} files")
    
    # Wait for conversion
    if not wait_for_conversion(session, session_id):
        print("⚠️  Some documents may not be converted, continuing anyway...")
    
    # Create run
    run_id = create_run(session, session_id, workflow_version_id)
    if not run_id:
        return
    
    # Wait for completion
    if not wait_for_run_completion(session, run_id):
        print("⚠️  Run may not have completed, checking outputs...")
    
    # Download outputs
    print("\n" + "="*60)
    print("Downloading Outputs")
    print("="*60)
    
    output_dir = Path("test_excel_outputs")
    output_dir.mkdir(exist_ok=True)
    
    for i, doc_id in enumerate(doc_ids, 1):
        output_path = output_dir / f"testcase_{i}_output.xlsx"
        download_output(session, run_id, doc_id, output_path)
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print(f"✓ Outputs saved in: {output_dir.absolute()}")
    print(f"✓ Run ID: {run_id}")

if __name__ == "__main__":
    main()

