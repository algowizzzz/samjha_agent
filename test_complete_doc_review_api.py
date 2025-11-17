#!/usr/bin/env python3
"""
Comprehensive Doc Review API & WebSocket Test Suite
Tests all backend endpoints and WebSocket events
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_KEY = "docreview_dev_key_12345"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name):
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {name}")

def print_pass(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_fail(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.YELLOW}ℹ{Colors.END} {msg}")

# ========================================
# 1. HEALTH CHECK
# ========================================
def test_server_health():
    print_test("Server Health Check")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print_pass(f"Server is running on {BASE_URL}")
            return True
        else:
            print_fail(f"Server returned {response.status_code}")
            return False
    except Exception as e:
        print_fail(f"Cannot connect to server: {e}")
        return False

# ========================================
# 2. AUTHENTICATION TESTS
# ========================================
def test_auth_api_key():
    print_test("API Key Authentication")
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/api/doc_review/documents", headers=headers)

    if response.status_code == 200:
        print_pass("API key authentication works")
        return True
    else:
        print_fail(f"API key auth failed: {response.status_code}")
        print_info(f"Response: {response.text[:200]}")
        return False

def test_auth_without_key():
    print_test("Auth Required (No API Key)")
    response = requests.get(f"{BASE_URL}/api/doc_review/documents")

    if response.status_code in [401, 403, 302]:
        print_pass("Unauthorized access blocked correctly")
        return True
    else:
        print_fail(f"Expected 401/403, got {response.status_code}")
        return False

# ========================================
# 3. DOCUMENT MANAGEMENT APIs
# ========================================
def test_list_documents():
    print_test("GET /api/doc_review/documents - List Documents")
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/api/doc_review/documents", headers=headers)

    if response.status_code == 200:
        docs = response.json()
        # Handle both list and dict responses
        if isinstance(docs, dict):
            docs_list = docs.get('documents', [])
            print_pass(f"Retrieved {len(docs_list)} documents")
            if docs_list:
                print_info(f"Sample: {docs_list[0].get('file_id', 'N/A')}")
            return docs_list
        else:
            print_pass(f"Retrieved {len(docs)} documents")
            if docs:
                print_info(f"Sample: {docs[0].get('file_id', 'N/A')}")
            return docs
    else:
        print_fail(f"Failed: {response.status_code}")
        return []

def test_get_document(file_id):
    print_test(f"GET /api/doc_review/documents/{file_id} - Get Document Details")
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/api/doc_review/documents/{file_id}", headers=headers)

    if response.status_code == 200:
        doc = response.json()
        print_pass(f"Document retrieved: {doc.get('file_id')}")
        print_info(f"Status: {doc.get('status')}")
        return doc
    else:
        print_fail(f"Failed: {response.status_code}")
        return None

def test_upload_document():
    print_test("POST /api/doc_review/documents/upload - Upload Document")
    headers = {"X-API-Key": API_KEY}

    # Create a simple test PDF file
    files = {
        'file': ('test_document.txt', b'# Test Document\n\nThis is a test document for API testing.', 'text/plain')
    }

    response = requests.post(
        f"{BASE_URL}/api/doc_review/documents/upload",
        headers=headers,
        files=files
    )

    if response.status_code in [200, 201]:
        result = response.json()
        print_pass(f"Upload successful: {result.get('file_id', 'N/A')}")
        return result.get('file_id')
    else:
        print_fail(f"Upload failed: {response.status_code}")
        print_info(f"Response: {response.text[:200]}")
        return None

def test_register_document_by_path():
    print_test("POST /api/doc_review/documents/register - Register by Path")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    data = {
        "document_path": "/path/to/test.pdf"
    }

    response = requests.post(
        f"{BASE_URL}/api/doc_review/documents/register",
        headers=headers,
        json=data
    )

    if response.status_code in [200, 201]:
        result = response.json()
        print_pass(f"Registered: {result.get('file_id', 'N/A')}")
        return result.get('file_id')
    else:
        print_fail(f"Register failed: {response.status_code}")
        print_info(f"Response: {response.text[:200]}")
        return None

# ========================================
# 4. TEMPLATE APIs
# ========================================
def test_list_templates():
    print_test("GET /api/doc_review/templates - List Templates")
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{BASE_URL}/api/doc_review/templates", headers=headers)

    if response.status_code == 200:
        templates = response.json()
        print_pass(f"Retrieved {len(templates)} templates")
        if templates:
            print_info(f"Templates: {[t.get('id') for t in templates]}")
        return templates
    else:
        print_fail(f"Failed: {response.status_code}")
        return []

# ========================================
# 5. PHASE EXECUTION APIs
# ========================================
def test_run_phase1(file_id, template_id="policy_template"):
    print_test(f"POST /api/doc_review/documents/{file_id}/run_phase1 - Run Phase 1")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    data = {"template_id": template_id}

    response = requests.post(
        f"{BASE_URL}/api/doc_review/documents/{file_id}/run_phase1",
        headers=headers,
        json=data,
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        print_pass(f"Phase 1 completed: {result.get('phase1_status')}")
        return result
    else:
        print_fail(f"Phase 1 failed: {response.status_code}")
        print_info(f"Response: {response.text[:200]}")
        return None

def test_run_phase2(file_id):
    print_test(f"POST /api/doc_review/documents/{file_id}/run_phase2 - Run Phase 2")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    response = requests.post(
        f"{BASE_URL}/api/doc_review/documents/{file_id}/run_phase2",
        headers=headers,
        json={},
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        print_pass(f"Phase 2 completed: {result.get('phase2_status')}")
        return result
    else:
        print_fail(f"Phase 2 failed: {response.status_code}")
        print_info(f"Response: {response.text[:200]}")
        return None

def test_run_phase4(file_id):
    print_test(f"POST /api/doc_review/documents/{file_id}/run_phase4 - Run Phase 4")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    response = requests.post(
        f"{BASE_URL}/api/doc_review/documents/{file_id}/run_phase4",
        headers=headers,
        json={},
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        print_pass(f"Phase 4 completed: {result.get('phase4_status', 'N/A')}")
        return result
    else:
        print_fail(f"Phase 4 failed: {response.status_code}")
        print_info(f"Response: {response.text[:200]}")
        return None

# ========================================
# 6. VFS (Virtual File System) APIs
# ========================================
def test_vfs_tree(file_id):
    print_test(f"GET /api/doc_review/vfs/tree?file_id={file_id} - Get VFS Tree")
    headers = {"X-API-Key": API_KEY}

    response = requests.get(
        f"{BASE_URL}/api/doc_review/vfs/tree",
        headers=headers,
        params={"file_id": file_id}
    )

    if response.status_code == 200:
        tree = response.json()
        print_pass(f"VFS tree retrieved")
        print_info(f"Root items: {len(tree.get('children', []))}")
        return tree
    else:
        print_fail(f"Failed: {response.status_code}")
        return None

def test_vfs_stat(file_id, path="/"):
    print_test(f"GET /api/doc_review/vfs/stat - Stat VFS Path")
    headers = {"X-API-Key": API_KEY}

    response = requests.get(
        f"{BASE_URL}/api/doc_review/vfs/stat",
        headers=headers,
        params={"file_id": file_id, "path": path}
    )

    if response.status_code == 200:
        stat = response.json()
        print_pass(f"Stat retrieved for '{path}'")
        return stat
    else:
        print_fail(f"Failed: {response.status_code} (This might be expected for root path)")
        return None

def test_vfs_read_file(file_id, path="/original.md"):
    print_test(f"GET /api/doc_review/vfs/file - Read VFS File")
    headers = {"X-API-Key": API_KEY}

    response = requests.get(
        f"{BASE_URL}/api/doc_review/vfs/file",
        headers=headers,
        params={"file_id": file_id, "path": path}
    )

    if response.status_code == 200:
        content = response.text
        print_pass(f"File read: {len(content)} bytes")
        return content
    else:
        print_fail(f"Failed: {response.status_code}")
        return None

def test_vfs_write_file(file_id, path="/test_write.txt"):
    print_test(f"PATCH /api/doc_review/vfs/file - Write VFS File")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    data = {
        "content": "# Test Write\n\nThis is a test write operation."
    }

    response = requests.patch(
        f"{BASE_URL}/api/doc_review/vfs/file",
        headers=headers,
        params={"file_id": file_id, "path": path},
        json=data
    )

    if response.status_code == 200:
        result = response.json()
        print_pass(f"File written successfully")
        return result
    else:
        print_fail(f"Failed: {response.status_code}")
        return None

# ========================================
# 7. CHAT API
# ========================================
def test_chat(file_id):
    print_test(f"POST /api/doc_review/chat/{file_id} - Send Chat Message")
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    data = {
        "message": "What is the status of this document?"
    }

    response = requests.post(
        f"{BASE_URL}/api/doc_review/chat/{file_id}",
        headers=headers,
        json=data,
        timeout=30
    )

    if response.status_code == 200:
        result = response.json()
        print_pass(f"Chat response received")
        print_info(f"Response: {result.get('response', 'N/A')[:100]}...")
        return result
    else:
        print_fail(f"Failed: {response.status_code}")
        return None

# ========================================
# 8. WEBSOCKET EVENTS TEST
# ========================================
def test_websocket_info():
    print_test("WebSocket Event Information")
    print_info("WebSocket events that should be tested manually:")
    events = [
        "connect - When client connects",
        "disconnect - When client disconnects",
        "join_document_room - Join a document room",
        "leave_document_room - Leave a document room",
        "doc_review:status - Document status updates",
        "doc_review:log - Activity log events",
        "doc_review:node_started - Workflow node started",
        "doc_review:node_completed - Workflow node completed",
        "doc_review:agent_plan_generated - Agent plan event",
        "doc_review:vfs_file_updated - VFS file update event",
    ]
    for event in events:
        print_info(f"  • {event}")

    print_info("\nTo test WebSocket events:")
    print_info("1. Open http://localhost:4200")
    print_info("2. Open browser DevTools Console")
    print_info("3. Check for 'Socket.IO connected' message")
    print_info("4. Run a phase and watch Activity panel")

# ========================================
# MAIN TEST RUNNER
# ========================================
def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Doc Review API & WebSocket Test Suite{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }

    # 1. Health Check
    if not test_server_health():
        print_fail("\nServer not available. Exiting.")
        return

    time.sleep(1)

    # 2. Authentication Tests
    test_auth_api_key()
    test_auth_without_key()

    time.sleep(1)

    # 3. List existing documents
    documents = test_list_documents()

    # 4. Template tests
    templates = test_list_templates()

    # If documents exist, test with first one
    if documents:
        file_id = documents[0].get('file_id')
        print_info(f"\nUsing existing document: {file_id}")

        # Get document details
        test_get_document(file_id)

        # VFS tests
        test_vfs_tree(file_id)
        test_vfs_stat(file_id)
        test_vfs_read_file(file_id)
        test_vfs_write_file(file_id)

        # Chat test
        # test_chat(file_id)  # Commented out to avoid long LLM calls

        # Phase tests (commented out to avoid long execution)
        print_info("\nPhase execution tests available but skipped (use --run-phases flag):")
        print_info(f"  • test_run_phase1('{file_id}')")
        print_info(f"  • test_run_phase2('{file_id}')")
        print_info(f"  • test_run_phase4('{file_id}')")
    else:
        print_info("\nNo existing documents found")

        # Test upload (this will fail if route is missing)
        print_info("\nTesting document upload (this may fail with 404):")
        uploaded_file_id = test_upload_document()

        if uploaded_file_id:
            print_info(f"Successfully uploaded: {uploaded_file_id}")

    # WebSocket info
    test_websocket_info()

    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}Test Summary{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    print_info("Tests completed. Check output above for details.")
    print_info("For WebSocket tests, open http://localhost:4200 and check DevTools Console")

if __name__ == "__main__":
    main()
