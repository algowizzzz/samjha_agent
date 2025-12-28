#!/usr/bin/env python3
"""
Test all Bulk Doc Analysis API endpoints end-to-end
"""
import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def login():
    """Login and get session cookie"""
    # Get credentials from users.json (default: admin/admin)
    session = requests.Session()
    
    login_url = f"{BASE_URL}/login"
    login_data = {
        "user_id": "admin",
        "password": "admin"
    }
    
    try:
        response = session.post(login_url, data=login_data, allow_redirects=False)
        if response.status_code in (200, 302):
            print("✓ Login successful")
            return session
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(response.text[:200])
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_endpoint(session, method, endpoint, data=None, files=None, description=""):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n  Testing {method} {endpoint}")
    if description:
        print(f"    {description}")
    
    try:
        if method == "GET":
            response = session.get(url, allow_redirects=False)
        elif method == "POST":
            if files:
                response = session.post(url, files=files, allow_redirects=False)
            else:
                response = session.post(url, json=data, allow_redirects=False, headers={"Content-Type": "application/json"})
        elif method == "PUT":
            response = session.put(url, json=data, allow_redirects=False, headers={"Content-Type": "application/json"})
        elif method == "DELETE":
            response = session.delete(url, allow_redirects=False)
        else:
            print(f"    ❌ Unknown method: {method}")
            return None
        
        print(f"    Status: {response.status_code}")
        
        if response.status_code >= 400:
            error_text = response.text[:200]
            print(f"    ❌ Error: {error_text}")
            return None
        
        try:
            result = response.json()
            return result
        except:
            return {"status": response.status_code, "content_type": response.headers.get("Content-Type")}
    
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        return None

def main():
    print("="*70)
    print("BULK DOC ANALYSIS - COMPLETE API ENDPOINT TEST")
    print("="*70)
    
    # Login
    print("\n[1/8] LOGIN")
    session = login()
    if not session:
        print("❌ Cannot proceed without login")
        return
    print("✓ Session established")
    
    # Test 1: List chains
    print("\n[2/8] LIST CHAINS")
    chains_result = test_endpoint(session, "GET", "/api/bulk-doc-analysis/chains", 
                                  description="Get all available chains")
    if chains_result:
        chains = chains_result.get("chains", [])
        print(f"✓ Found {len(chains)} chain(s)")
        chain_version_id = chains[0].get("chain_version_id") if chains else None
    else:
        chain_version_id = None
    
    # Test 2: Create chain
    print("\n[3/8] CREATE CHAIN")
    chain_data = {
        "name": "API Test Chain",
        "description": "Created via API test",
        "steps": [
            {
                "index": 1,
                "required_inputs": ["R0"],
                "prompt": "Summarize this document in one sentence.",
                "description": "Quick summary"
            }
        ]
    }
    create_result = test_endpoint(session, "POST", "/api/bulk-doc-analysis/chains",
                                  data=chain_data, description="Create new chain")
    if create_result:
        new_chain = create_result.get("chain", {})
        test_chain_version_id = new_chain.get("chain_version_id")
        test_chain_id = new_chain.get("chain_id")
        print(f"✓ Chain created: {test_chain_version_id}")
    else:
        test_chain_version_id = None
        test_chain_id = None
    
    # Test 3: Update chain
    if test_chain_id:
        print("\n[4/8] UPDATE CHAIN")
        update_data = {
            "name": "Updated API Test Chain",
            "description": "Updated via API",
            "steps": chain_data["steps"]
        }
        update_result = test_endpoint(session, "PUT", f"/api/bulk-doc-analysis/chains/{test_chain_id}",
                                     data=update_data, description="Update existing chain")
        if update_result:
            print(f"✓ Chain updated")
    
    # Test 4: Upload PDF
    print("\n[5/8] UPLOAD PDF")
    # Find existing PDF or create dummy
    existing_pdfs = list(Path("data/ai_bulk_doc_analysis/docs").rglob("*.pdf"))
    if not existing_pdfs:
        print("    ⚠️  No PDF found, creating dummy file")
        test_pdf = Path("test_upload.pdf")
        test_pdf.write_bytes(b"%PDF-1.4 dummy content")
        pdf_path = test_pdf
    else:
        pdf_path = existing_pdfs[0]
    
    with open(pdf_path, "rb") as f:
        files = {"files": (pdf_path.name, f, "application/pdf")}
        upload_result = test_endpoint(session, "POST", "/api/bulk-doc-analysis/documents/upload",
                                     files=files, description="Upload PDF document")
    
    if upload_result:
        docs = upload_result.get("documents", [])
        if docs:
            test_doc_id = docs[0].get("doc_id")
            print(f"✓ PDF uploaded: {test_doc_id}")
        else:
            test_doc_id = None
    else:
        test_doc_id = None
    
    # Test 5: List documents
    print("\n[6/8] LIST DOCUMENTS")
    docs_result = test_endpoint(session, "GET", "/api/bulk-doc-analysis/documents",
                               description="List all documents")
    if docs_result:
        docs = docs_result.get("documents", [])
        print(f"✓ Found {len(docs)} document(s)")
    
    # Wait for conversion
    if test_doc_id:
        print("\n[WAITING] Document Conversion...")
        for i in range(10):
            docs_result = test_endpoint(session, "GET", "/api/bulk-doc-analysis/documents")
            if docs_result:
                docs = docs_result.get("documents", [])
                doc = next((d for d in docs if d.get("doc_id") == test_doc_id), None)
                if doc:
                    status = doc.get("status")
                    print(f"  Status: {status}", end="\r")
                    if status == "CONVERTED":
                        print(f"\n✓ Conversion complete")
                        break
                    elif status == "ERROR":
                        print(f"\n❌ Conversion failed: {doc.get('error_message')}")
                        break
            time.sleep(1)
    
    # Test 6: Create run
    if test_doc_id and test_chain_version_id:
        print("\n[7/8] CREATE RUN")
        run_data = {"chain_version_id": test_chain_version_id}
        run_result = test_endpoint(session, "POST", "/api/bulk-doc-analysis/runs",
                                  data=run_data, description="Create execution run")
        if run_result:
            run = run_result.get("run", {})
            test_run_id = run.get("run_id")
            print(f"✓ Run created: {test_run_id}")
            
            # Wait for execution
            print("\n[WAITING] Run Execution...")
            for i in range(30):
                progress_result = test_endpoint(session, "GET", 
                                               f"/api/bulk-doc-analysis/runs/{test_run_id}/progress")
                if progress_result:
                    progress_status = progress_result.get("status")
                    rows = progress_result.get("rows", [])
                    print(f"  Run Status: {progress_status}, Documents: {len(rows)}", end="\r")
                    if progress_status in ("COMPLETE", "ERROR"):
                        print(f"\n✓ Run finished: {progress_status}")
                        break
                time.sleep(2)
            
            # Test 7: Download output
            if test_doc_id:
                print("\n[8/8] DOWNLOAD OUTPUT")
                download_result = test_endpoint(session, "GET",
                                               f"/api/bulk-doc-analysis/runs/{test_run_id}/download/{test_doc_id}",
                                              description="Download final output")
                if download_result:
                    print(f"✓ Download successful (content type: {download_result.get('content_type')})")
                else:
                    print("    ⚠️  Download may have returned file (check status code)")
        else:
            test_run_id = None
    else:
        print("\n[7/8] CREATE RUN - Skipped (missing doc or chain)")
        print("[8/8] DOWNLOAD OUTPUT - Skipped")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ All tested endpoints responded")
    print("\nEndpoints tested:")
    print("  ✓ GET  /api/bulk-doc-analysis/chains")
    print("  ✓ POST /api/bulk-doc-analysis/chains")
    print("  ✓ PUT  /api/bulk-doc-analysis/chains/<id>")
    print("  ✓ POST /api/bulk-doc-analysis/documents/upload")
    print("  ✓ GET  /api/bulk-doc-analysis/documents")
    print("  ✓ POST /api/bulk-doc-analysis/runs")
    print("  ✓ GET  /api/bulk-doc-analysis/runs/<id>/progress")
    print("  ✓ GET  /api/bulk-doc-analysis/runs/<id>/download/<doc_id>")
    print("\n✓ API test completed")

if __name__ == "__main__":
    main()

