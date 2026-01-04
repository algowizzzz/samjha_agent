#!/usr/bin/env python3
"""
End-to-end test for AI Bulk Doc Analysis APIs.
Tests: Create chain → Upload PDF → Run → Download output
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
SESSION_COOKIE = None  # Will be set after login

def check_auth():
    """Check if we can access APIs (tests auth)"""
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains"
    try:
        response = requests.get(url, allow_redirects=False, timeout=5)
        if response.status_code == 401:
            print("❌ Not authenticated. Please login via browser first.")
            print("   Then run this test - it will use your session cookie.")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"⚠️  Auth check error: {e}")
        return True  # Continue anyway

def create_test_pdf():
    """Create a simple test PDF file"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        print("reportlab not available, trying alternative...")
        # Create a simple text file as placeholder
        pdf_path = Path("test_document.pdf")
        with open(pdf_path, "w") as f:
            f.write("This is a test document.\n" * 10)
        print(f"Created text file as placeholder: {pdf_path}")
        return pdf_path
    
    pdf_path = Path("test_document.pdf")
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Test Document for AI Bulk Doc Analysis")
    c.drawString(100, 730, "This is a sample PDF document.")
    c.drawString(100, 710, "It contains multiple paragraphs of text.")
    c.drawString(100, 690, "The system should convert this to Markdown.")
    c.drawString(100, 670, "Then analyze it using prompt chains.")
    c.drawString(100, 650, "This is the second paragraph.")
    c.drawString(100, 630, "It demonstrates multi-paragraph content.")
    c.showPage()
    c.save()
    print(f"✓ Created test PDF: {pdf_path}")
    return pdf_path

def get_session_cookies():
    """Get session cookies by logging in"""
    login_url = f"{BASE_URL}/login"
    login_data = {"user_id": "admin", "password": "admin123"}
    
    try:
        session = requests.Session()
        # Login uses form-based POST
        response = session.post(login_url, data=login_data, allow_redirects=True)
        # Check if we got redirected (success) or stayed on login (failure)
        if response.status_code == 200 and "/login" not in response.url:
            print("✓ Login successful")
            return session
        elif response.status_code == 200:
            # Check response text for error
            if "Invalid credentials" in response.text or "error" in response.text.lower():
                print(f"⚠️  Login failed: Invalid credentials")
            else:
                print(f"⚠️  Login response unclear: {response.status_code}")
            return None
        else:
            print(f"⚠️  Login failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️  Login error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_create_chain(session=None):
    """Test: Create a new prompt chain"""
    print("\n" + "="*60)
    print("TEST 1: Create Chain (POST /api/bulk-doc-analysis/chains)")
    print("="*60)
    
    chain_data = {
        "name": "Test 2-Step Analysis Chain",
        "description": "Simple test chain for end-to-end testing",
        "steps": [
            {
                "index": 1,
                "title": "Summarize Document",
                "required_inputs": ["R0"],
                "prompt": "Summarize this document in 2-3 sentences. Focus on the main topics discussed.",
                "description": "Initial summary",
                "model_config": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "temperature": 0.2
                }
            },
            {
                "index": 2,
                "title": "Extract Themes",
                "required_inputs": ["R0", "R1"],
                "prompt": "Based on the document and the summary, identify any key themes or important points. Present them as a bulleted list.",
                "description": "Theme extraction",
                "model_config": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "temperature": 0.2
                }
            }
        ]
    }
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains"
    headers = {"Content-Type": "application/json"}
    
    try:
        if session:
            response = session.post(url, json=chain_data, headers=headers)
        else:
            response = requests.post(url, json=chain_data, headers=headers, allow_redirects=False)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED - Need to login first")
            return None
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        if not result.get("success"):
            print(f"❌ API returned success=false: {result}")
            return None
            
        chain = result.get("chain", {})
        chain_version_id = chain.get("chain_version_id")
        chain_id = chain.get("chain_id")
        
        print(f"✓ Chain created successfully")
        print(f"  Chain ID: {chain_id}")
        print(f"  Chain Version ID: {chain_version_id}")
        print(f"  Name: {chain.get('name')}")
        print(f"  Steps: {chain.get('step_count')}")
        
        return {"chain_id": chain_id, "chain_version_id": chain_version_id}
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_list_chains(session=None):
    """Test: List all chains"""
    print("\n" + "="*60)
    print("TEST 2: List Chains (GET /api/bulk-doc-analysis/chains)")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains"
    
    try:
        if session:
            response = session.get(url)
        else:
            response = requests.get(url, allow_redirects=False)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return []
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return []
        
        result = response.json()
        chains = result.get("chains", [])
        
        print(f"✓ Found {len(chains)} chain(s)")
        for chain in chains[:5]:  # Show first 5
            print(f"  - {chain.get('name')} ({chain.get('chain_version_id')[:20]}...)")
        if len(chains) > 5:
            print(f"  ... and {len(chains) - 5} more")
        
        return chains
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_get_chain(chain_id, session=None):
    """Test: Get chain by ID"""
    print("\n" + "="*60)
    print(f"TEST 3: Get Chain (GET /api/bulk-doc-analysis/chains/{chain_id})")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains/{chain_id}"
    
    try:
        if session:
            response = session.get(url)
        else:
            response = requests.get(url, allow_redirects=False)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return None
        
        if response.status_code == 404:
            print("❌ Chain not found")
            return None
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None
        
        chain = response.json()
        
        print(f"✓ Chain retrieved successfully")
        print(f"  Chain ID: {chain.get('chain_id')}")
        print(f"  Name: {chain.get('name')}")
        print(f"  Steps: {chain.get('step_count')}")
        print(f"  Valid: {chain.get('valid')}")
        
        return chain
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_update_chain(chain_id, session=None):
    """Test: Update chain (creates new version)"""
    print("\n" + "="*60)
    print(f"TEST 4: Update Chain (PUT /api/bulk-doc-analysis/chains/{chain_id})")
    print("="*60)
    
    chain_data = {
        "name": "Test 2-Step Analysis Chain (Updated)",
        "description": "Updated description for testing",
        "steps": [
            {
                "index": 1,
                "title": "Summarize Document",
                "required_inputs": ["R0"],
                "prompt": "Summarize this document in 2-3 sentences. Focus on the main topics discussed.",
                "description": "Initial summary",
                "model_config": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "temperature": 0.2
                }
            },
            {
                "index": 2,
                "title": "Extract Themes",
                "required_inputs": ["R0", "R1"],
                "prompt": "Based on the document and the summary, identify any key themes or important points. Present them as a bulleted list.",
                "description": "Theme extraction",
                "model_config": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "temperature": 0.2
                }
            },
            {
                "index": 3,
                "title": "Key Insights",
                "required_inputs": ["R1", "R2"],
                "prompt": "Extract 3 key insights from the themes identified.",
                "description": "Insight extraction",
                "model_config": {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 2048,
                    "temperature": 0.3
                }
            }
        ]
    }
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains/{chain_id}"
    headers = {"Content-Type": "application/json"}
    
    try:
        if session:
            response = session.put(url, json=chain_data, headers=headers)
        else:
            response = requests.put(url, json=chain_data, headers=headers, allow_redirects=False)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return None
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None
        
        result = response.json()
        if not result.get("success"):
            print(f"❌ API returned success=false: {result}")
            return None
            
        chain = result.get("chain", {})
        new_chain_version_id = chain.get("chain_version_id")
        
        print(f"✓ Chain updated successfully (new version created)")
        print(f"  Chain ID: {chain.get('chain_id')}")
        print(f"  New Chain Version ID: {new_chain_version_id}")
        print(f"  Steps: {chain.get('step_count')} (added 1 new step)")
        
        return new_chain_version_id
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_delete_chain(chain_id, session=None):
    """Test: Delete chain"""
    print("\n" + "="*60)
    print(f"TEST 5: Delete Chain (DELETE /api/bulk-doc-analysis/chains/{chain_id})")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains/{chain_id}"
    
    try:
        if session:
            response = session.delete(url)
        else:
            response = requests.delete(url, allow_redirects=False)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return False
        
        if response.status_code == 404:
            print("❌ Chain not found")
            return False
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
        
        result = response.json()
        if not result.get("success"):
            print(f"❌ API returned success=false: {result}")
            return False
        
        print(f"✓ Chain deleted successfully")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_create_run_with_workflow_version_id(session=None):
    """Test: Create run using workflow_version_id (new scalable approach)"""
    print("\n" + "="*60)
    print("TEST: Create Run with workflow_version_id")
    print("="*60)
    
    # First, we need to create a workflow to test with
    # This test assumes we have a workflow already created
    # For now, just verify the API accepts workflow_version_id
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs"
    data = {
        "workflow_version_id": "test_workflow_version_id"  # This will fail validation but tests the endpoint
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        if session:
            response = session.post(url, json=data, headers=headers)
        else:
            response = requests.post(url, json=data, headers=headers, allow_redirects=False)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return False
        
        # We expect 400 or 404 (invalid workflow_version_id), but not 400 saying chain_version_id is required
        if response.status_code == 400:
            error_text = response.text
            if "chain_version_id" in error_text and "required" in error_text.lower():
                print("❌ API still requires chain_version_id (backend fix not working)")
                print(f"Response: {error_text[:200]}")
                return False
            else:
                print(f"✓ API accepts workflow_version_id (got expected validation error)")
                print(f"  Response: {error_text[:200]}")
                return True
        
        print(f"✓ API accepts workflow_version_id parameter")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_upload_pdf(pdf_path, session=None):
    """Test: Upload PDF document"""
    print("\n" + "="*60)
    print("TEST 5: Upload PDF")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/documents/upload"
    
    try:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            if session:
                response = session.post(url, files=files)
            else:
                response = requests.post(url, files=files, allow_redirects=False)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return None
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        documents = result.get("documents", [])
        
        if not documents:
            print("❌ No documents returned")
            return None
        
        doc = documents[0]
        doc_id = doc.get("doc_id")
        
        print(f"✓ PDF uploaded successfully")
        print(f"  Doc ID: {doc_id}")
        print(f"  Filename: {doc.get('original_filename')}")
        print(f"  Status: {doc.get('status')}")
        
        return doc_id
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_list_documents(session=None):
    """Test: List documents"""
    print("\n" + "="*60)
    print("TEST 6: List Documents")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/documents"
    
    try:
        if session:
            response = session.get(url)
        else:
            response = requests.get(url, allow_redirects=False)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return []
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            return []
        
        result = response.json()
        documents = result.get("documents", [])
        
        print(f"✓ Found {len(documents)} document(s)")
        for doc in documents:
            print(f"  - {doc.get('original_filename')} ({doc.get('status')})")
        
        return documents
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return []

def wait_for_conversion(doc_id, session=None, timeout=30):
    """Wait for document conversion to complete"""
    print("\n" + "="*60)
    print("WAITING: Document Conversion")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/documents"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            if session:
                response = session.get(url)
            else:
                response = requests.get(url, allow_redirects=False)
            if response.status_code == 200:
                result = response.json()
                documents = result.get("documents", [])
                doc = next((d for d in documents if d.get("doc_id") == doc_id), None)
                
                if doc:
                    status = doc.get("status")
                    print(f"  Status: {status}", end="\r")
                    
                    if status == "CONVERTED":
                        print(f"\n✓ Conversion complete")
                        return True
                    elif status == "ERROR":
                        print(f"\n❌ Conversion failed")
                        print(f"   Error: {doc.get('error_message')}")
                        return False
                
            time.sleep(1)
        except Exception as e:
            print(f"\n❌ Error checking status: {e}")
            return False
    
    print(f"\n⏱️  Timeout waiting for conversion")
    return False

def test_create_run(chain_version_id, session=None):
    """Test: Create a run"""
    print("\n" + "="*60)
    print("TEST 7: Create Run")
    print("="*60)
    
    # Get session_id (we'll use a dummy one - in real app this comes from auth)
    # The API should handle this via session cookie
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs"
    data = {
        "chain_version_id": chain_version_id
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        if session:
            response = session.post(url, json=data, headers=headers)
        else:
            response = requests.post(url, json=data, headers=headers, allow_redirects=False)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return None
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        result = response.json()
        run = result.get("run", {})
        run_id = run.get("run_id")
        
        print(f"✓ Run created successfully")
        print(f"  Run ID: {run_id}")
        print(f"  Status: {run.get('status')}")
        
        return run_id
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def wait_for_run_completion(run_id, session=None, timeout=120):
    """Wait for run execution to complete"""
    print("\n" + "="*60)
    print("WAITING: Run Execution")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/progress"
    start_time = time.time()
    last_status = {}
    
    while time.time() - start_time < timeout:
        try:
            if session:
                response = session.get(url)
            else:
                response = requests.get(url, allow_redirects=False)
            if response.status_code == 200:
                progress = response.json()
                rows = progress.get("rows", [])
                run_status = progress.get("status")
                
                # Print progress
                status_summary = {}
                for row in rows:
                    status = row.get("status", "UNKNOWN")
                    status_summary[status] = status_summary.get(status, 0) + 1
                
                if status_summary != last_status:
                    print(f"  Run Status: {run_status}")
                    for status, count in status_summary.items():
                        print(f"    {status}: {count}")
                    last_status = status_summary
                    print()
                
                if run_status in ("COMPLETE", "ERROR"):
                    print(f"✓ Run finished with status: {run_status}")
                    return progress
                
            time.sleep(2)
        except Exception as e:
            print(f"\n❌ Error checking progress: {e}")
            return None
    
    print(f"\n⏱️  Timeout waiting for run completion")
    return None

def test_download_output(run_id, doc_id, session=None):
    """Test: Download final output"""
    print("\n" + "="*60)
    print("TEST 8: Download Output")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/download/{doc_id}"
    
    try:
        if session:
            response = session.get(url)
        else:
            response = requests.get(url, allow_redirects=False)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return False
        
        if response.status_code == 404:
            print("❌ Output not found")
            return False
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        print(f"✓ Download successful")
        print(f"  Content-Type: {content_type}")
        
        # Save file
        output_path = Path(f"downloaded_output_{run_id}.md")
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        # Show preview
        content = response.content.decode("utf-8", errors="ignore")
        preview = content[:500] + "..." if len(content) > 500 else content
        print(f"  Saved to: {output_path}")
        print(f"  Size: {len(content)} characters")
        print(f"\n  Preview:\n{'-'*40}")
        print(preview)
        print(f"{'-'*40}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("AI BULK DOC ANALYSIS - END-TO-END API TEST")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print("✓ Server is running")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print(f"   Start server with: python3 run_server.py")
        return
    except:
        pass
    
    # Try to get authenticated session
    print("\nAttempting to login...")
    session = get_session_cookies()
    if not session:
        print("⚠️  Could not login automatically. Some tests may fail with 401.")
        print("   You can login via browser first, then run tests.")
        session = None
    else:
        print("✓ Logged in successfully")
    
    print()
    
    # Create test PDF
    pdf_path = create_test_pdf()
    if not pdf_path or not pdf_path.exists():
        print("❌ Could not create test PDF")
        return
    
    # Test sequence
    issues = []
    
    # 1. Create chain
    chain_result = test_create_chain(session)
    if not chain_result:
        issues.append("Failed to create chain")
        print("\n❌ Cannot continue without chain")
        return
    
    chain_id = chain_result["chain_id"]
    chain_version_id = chain_result["chain_version_id"]
    
    # 2. List chains (verify)
    chains = test_list_chains(session)
    if not any(c.get("chain_id") == chain_id for c in chains):
        issues.append("Created chain not found in list")
    
    # 2b. Get chain by ID
    retrieved_chain = test_get_chain(chain_id, session)
    if not retrieved_chain:
        issues.append("Failed to retrieve chain by ID")
    
    # 2c. Update chain (creates new version)
    new_chain_version_id = test_update_chain(chain_id, session)
    if not new_chain_version_id:
        issues.append("Failed to update chain")
    
    # 2d. Test run creation with workflow_version_id
    test_create_run_with_workflow_version_id(session)
    
    # Note: We skip delete test to keep the chain for workflow tests
    # Uncomment if you want to test delete:
    # test_delete_chain(chain_id, session)
    
    # 5. Upload PDF
    doc_id = test_upload_pdf(pdf_path, session)
    if not doc_id:
        issues.append("Failed to upload PDF")
        print("\n❌ Cannot continue without document")
        return
    
    # 6. List documents (verify)
    docs = test_list_documents(session)
    if not any(d.get("doc_id") == doc_id for d in docs):
        issues.append("Uploaded document not found in list")
    
    # 7. Wait for conversion
    if not wait_for_conversion(doc_id, session):
        issues.append("Document conversion failed or timed out")
        print("\n❌ Cannot continue without converted document")
        return
    
    # 8. Create run
    run_id = test_create_run(chain_version_id, session)
    if not run_id:
        issues.append("Failed to create run")
        print("\n❌ Cannot continue without run")
        return
    
    # 9. Wait for run completion
    progress = wait_for_run_completion(run_id, session)
    if not progress:
        issues.append("Run execution failed or timed out")
    
    # 10. Download output
    if progress:
        rows = progress.get("rows", [])
        if rows:
            success_row = next((r for r in rows if r.get("status") == "SUCCESS"), None)
            if success_row:
                test_doc_id = success_row.get("doc_id", doc_id)
                if not test_download_output(run_id, test_doc_id, session):
                    issues.append("Failed to download output")
            else:
                issues.append("No successful documents to download")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if issues:
        print(f"❌ Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All tests passed!")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()

