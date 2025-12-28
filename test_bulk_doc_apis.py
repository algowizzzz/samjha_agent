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

def test_create_chain():
    """Test: Create a new prompt chain"""
    print("\n" + "="*60)
    print("TEST 1: Create Chain")
    print("="*60)
    
    chain_data = {
        "name": "Test 2-Step Analysis Chain",
        "description": "Simple test chain for end-to-end testing",
        "steps": [
            {
                "index": 1,
                "required_inputs": ["R0"],
                "prompt": "Summarize this document in 2-3 sentences. Focus on the main topics discussed.",
                "description": "Initial summary"
            },
            {
                "index": 2,
                "required_inputs": ["R0", "R1"],
                "prompt": "Based on the document and the summary, identify any key themes or important points. Present them as a bulleted list.",
                "description": "Theme extraction"
            }
        ]
    }
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains"
    headers = {"Content-Type": "application/json"}
    
    try:
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
        chain = result.get("chain", {})
        chain_version_id = chain.get("chain_version_id")
        chain_id = chain.get("chain_id")
        
        print(f"✓ Chain created successfully")
        print(f"  Chain ID: {chain_id}")
        print(f"  Chain Version ID: {chain_version_id}")
        print(f"  Name: {chain.get('name')}")
        print(f"  Steps: {chain.get('step_count')}")
        
        return chain_version_id
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_list_chains():
    """Test: List all chains"""
    print("\n" + "="*60)
    print("TEST 2: List Chains")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/chains"
    
    try:
        response = requests.get(url, allow_redirects=False)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 401:
            print("❌ UNAUTHORIZED")
            return []
        
        if response.status_code != 200:
            print(f"❌ FAILED: {response.status_code}")
            return []
        
        result = response.json()
        chains = result.get("chains", [])
        
        print(f"✓ Found {len(chains)} chain(s)")
        for chain in chains:
            print(f"  - {chain.get('name')} ({chain.get('chain_version_id')})")
        
        return chains
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return []

def test_upload_pdf(pdf_path):
    """Test: Upload PDF document"""
    print("\n" + "="*60)
    print("TEST 3: Upload PDF")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/documents/upload"
    
    try:
        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
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

def test_list_documents():
    """Test: List documents"""
    print("\n" + "="*60)
    print("TEST 4: List Documents")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/documents"
    
    try:
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

def wait_for_conversion(doc_id, timeout=30):
    """Wait for document conversion to complete"""
    print("\n" + "="*60)
    print("WAITING: Document Conversion")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/documents"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
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

def test_create_run(chain_version_id):
    """Test: Create a run"""
    print("\n" + "="*60)
    print("TEST 5: Create Run")
    print("="*60)
    
    # Get session_id (we'll use a dummy one - in real app this comes from auth)
    # The API should handle this via session cookie
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs"
    data = {
        "chain_version_id": chain_version_id
    }
    headers = {"Content-Type": "application/json"}
    
    try:
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

def wait_for_run_completion(run_id, timeout=120):
    """Wait for run execution to complete"""
    print("\n" + "="*60)
    print("WAITING: Run Execution")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/progress"
    start_time = time.time()
    last_status = {}
    
    while time.time() - start_time < timeout:
        try:
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

def test_download_output(run_id, doc_id):
    """Test: Download final output"""
    print("\n" + "="*60)
    print("TEST 6: Download Output")
    print("="*60)
    
    url = f"{BASE_URL}/api/bulk-doc-analysis/runs/{run_id}/download/{doc_id}"
    
    try:
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
    
    # Check if server is running and we're authenticated
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print("✓ Server is running")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print(f"   Start server with: bash start_server.sh")
        return
    except:
        pass
    
    if not check_auth():
        return
    print()
    
    # Create test PDF
    pdf_path = create_test_pdf()
    if not pdf_path or not pdf_path.exists():
        print("❌ Could not create test PDF")
        return
    
    # Test sequence
    issues = []
    
    # 1. Create chain
    chain_version_id = test_create_chain()
    if not chain_version_id:
        issues.append("Failed to create chain")
        print("\n❌ Cannot continue without chain")
        return
    
    # 2. List chains (verify)
    chains = test_list_chains()
    if not any(c.get("chain_version_id") == chain_version_id for c in chains):
        issues.append("Created chain not found in list")
    
    # 3. Upload PDF
    doc_id = test_upload_pdf(pdf_path)
    if not doc_id:
        issues.append("Failed to upload PDF")
        print("\n❌ Cannot continue without document")
        return
    
    # 4. List documents (verify)
    docs = test_list_documents()
    if not any(d.get("doc_id") == doc_id for d in docs):
        issues.append("Uploaded document not found in list")
    
    # 5. Wait for conversion
    if not wait_for_conversion(doc_id):
        issues.append("Document conversion failed or timed out")
        print("\n❌ Cannot continue without converted document")
        return
    
    # 6. Create run
    run_id = test_create_run(chain_version_id)
    if not run_id:
        issues.append("Failed to create run")
        print("\n❌ Cannot continue without run")
        return
    
    # 7. Wait for run completion
    progress = wait_for_run_completion(run_id)
    if not progress:
        issues.append("Run execution failed or timed out")
    
    # 8. Download output
    if progress:
        rows = progress.get("rows", [])
        if rows:
            success_row = next((r for r in rows if r.get("status") == "SUCCESS"), None)
            if success_row:
                test_doc_id = success_row.get("doc_id", doc_id)
                if not test_download_output(run_id, test_doc_id):
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

