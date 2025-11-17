"""
Simplified test script for Model Documentation Agent API endpoints.
This version uses manual session setup instructions if automatic auth fails.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

import requests

# Configuration
BASE_URL = "http://localhost:5000"
API_BASE = f"{BASE_URL}/api/model_doc"


def check_server() -> bool:
    """Check if server is accessible."""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        return response.status_code < 500
    except:
        return False


def create_sample_codebase(base_dir: Path) -> Path:
    """Create a sample codebase for testing."""
    codebase_dir = base_dir / "test_sample_codebase"
    codebase_dir.mkdir(exist_ok=True)
    
    (codebase_dir / "__init__.py").write_text('"""Sample codebase for testing."""\n')
    
    (codebase_dir / "calculator.py").write_text('''"""
Calculator module with basic arithmetic operations.
"""


class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
''')
    
    (codebase_dir / "utils").mkdir(exist_ok=True)
    (codebase_dir / "utils" / "__init__.py").write_text('"""Utility functions."""\n')
    
    (codebase_dir / "utils" / "helpers.py").write_text('''"""
Helper utility functions.
"""


def format_number(num: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{num:.{decimals}f}"


def validate_input(value: str) -> float:
    """Validate and convert input string to float."""
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Invalid number: {value}")
''')
    
    print(f"✅ Created sample codebase at: {codebase_dir}")
    return codebase_dir


def test_with_session(session: requests.Session, codebase_path: Path):
    """Test all endpoints with an authenticated session."""
    print("\n" + "=" * 70)
    print("Testing API Endpoints")
    print("=" * 70)
    
    # Test 1: List templates
    print("\n1️⃣  Testing: GET /api/model_doc/templates")
    r = session.get(f"{API_BASE}/templates")
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        templates = data.get("templates", [])
        print(f"   ✅ Found {len(templates)} templates")
    else:
        print(f"   ❌ Error: {r.text[:100]}")
    
    # Test 2: Register codebase
    print("\n2️⃣  Testing: POST /api/model_doc/documents")
    payload = {
        "codebase_path": str(codebase_path),
        "codebase_id": "test_calculator_codebase"
    }
    r = session.post(f"{API_BASE}/documents", json=payload)
    print(f"   Status: {r.status_code}")
    if r.status_code in [200, 201]:
        data = r.json()
        codebase_id = data.get("codebase", {}).get("codebase_id", "test_calculator_codebase")
        print(f"   ✅ Registered codebase: {codebase_id}")
        
        # Test 3: List codebases
        print("\n3️⃣  Testing: GET /api/model_doc/documents")
        r = session.get(f"{API_BASE}/documents")
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            codebases = data.get("codebases", [])
            print(f"   ✅ Found {len(codebases)} codebases")
        
        # Test 4: Get codebase
        print(f"\n4️⃣  Testing: GET /api/model_doc/documents/{codebase_id}")
        r = session.get(f"{API_BASE}/documents/{codebase_id}")
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            codebase = data.get("codebase", {})
            print(f"   ✅ Retrieved codebase: {codebase.get('codebase_id')}")
            print(f"      Status: {codebase.get('status')}")
        
        # Test 5: Get status
        print(f"\n5️⃣  Testing: GET /api/model_doc/documents/{codebase_id}/status")
        r = session.get(f"{API_BASE}/documents/{codebase_id}/status")
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   ✅ Status: {data.get('status')}")
        
        # Test 6: Update config
        print(f"\n6️⃣  Testing: PATCH /api/model_doc/documents/{codebase_id}/config")
        r = session.patch(
            f"{API_BASE}/documents/{codebase_id}/config",
            json={"config": {"llm": {"temperature": 0.3}}}
        )
        print(f"   Status: {r.status_code}")
        if r.status_code in [200, 204]:
            print(f"   ✅ Config updated")
        
        # Test 7: Run Phase 1
        print(f"\n7️⃣  Testing: POST /api/model_doc/documents/{codebase_id}/run_phase1")
        print("   ⏳ Running Phase 1 workflow...")
        r = session.post(f"{API_BASE}/documents/{codebase_id}/run_phase1", json={})
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            codebase = data.get("codebase", {})
            state = codebase.get("state", {})
            file_list = state.get("file_list", [])
            print(f"   ✅ Phase 1 completed")
            print(f"      Files discovered: {len(file_list)}")
            metadata = state.get("codebase_metadata", {})
            if metadata:
                print(f"      Total files: {metadata.get('file_count', 0)}")
                stats = state.get("code_stats", {})
                print(f"      Classes: {stats.get('total_classes', 0)}")
                print(f"      Functions: {stats.get('total_functions', 0)}")
                print(f"      Lines of code: {stats.get('total_lines', 0)}")
        else:
            print(f"   ❌ Error: {r.text[:200]}")
        
        # Test 8: Chat (may fail if LLM not configured)
        print(f"\n8️⃣  Testing: POST /api/model_doc/chat/{codebase_id}")
        r = session.post(
            f"{API_BASE}/chat/{codebase_id}",
            json={"message": "What classes are in this codebase?"}
        )
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            response = data.get("response", "")
            print(f"   ✅ Chat response received: {len(response)} characters")
        else:
            print(f"   ⚠️  Chat failed (may require LLM): {r.status_code}")
        
        print("\n" + "=" * 70)
        print("✅ All endpoint tests completed!")
        print("=" * 70)
        
        return codebase_id
    else:
        print(f"   ❌ Error: {r.text[:200]}")
        return None


def main():
    print("=" * 70)
    print("Model Documentation Agent - API Endpoint Tests")
    print("=" * 70)
    
    # Check server
    if not check_server():
        print(f"\n❌ Server is not running or not accessible at {BASE_URL}")
        print(f"   Please start the server first:")
        print(f"      python3 run_server.py")
        return
    print(f"\n✅ Server is running at {BASE_URL}")
    
    # Create sample codebase
    test_data_dir = Path(__file__).parent.parent / "data" / "model_doc"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    codebase_path = create_sample_codebase(test_data_dir)
    
    # Create session
    session = requests.Session()
    
    # Instructions for manual authentication
    print("\n" + "=" * 70)
    print("AUTHENTICATION SETUP REQUIRED")
    print("=" * 70)
    print("\nTo test the API endpoints, you need an authenticated session.")
    print("\nOption 1: Manual Browser Login (Recommended)")
    print("   1. Open a web browser")
    print(f"   2. Navigate to: {BASE_URL}/login")
    print("   3. Login with:")
    print(f"      User ID: admin")
    print(f"      Password: admin123")
    print("   4. After login, copy your session cookies from browser DevTools")
    print("      (Look for cookies named 'session' or 'token')")
    print("\nOption 2: Use Browser Session")
    print("   1. Login via browser at the URL above")
    print("   2. Copy cookies from browser")
    print("   3. Set them in this script")
    print("\nOption 3: Disable Auth for Testing")
    print("   Modify routes to temporarily allow unauthenticated access")
    print("\n" + "=" * 70)
    
    # Try web login
    print("\n🔐 Attempting authentication...")
    try:
        # Get login page first (sets session cookie)
        session.get(f"{BASE_URL}/login")
        
        # Try login
        r = session.post(
            f"{BASE_URL}/login",
            data={"user_id": "admin", "password": "admin123"},
            allow_redirects=False
        )
        
        if r.status_code == 302 or any('session' in str(c).lower() for c in session.cookies):
            print("   ✅ Authentication successful via web login!")
            test_with_session(session, codebase_path)
        else:
            print("   ⚠️  Web login failed. Please login manually in browser and copy cookies.")
            print("\n   You can manually test endpoints using curl or Postman:")
            print(f"   curl -b 'session=<your_session_cookie>' {API_BASE}/templates")
    except Exception as e:
        print(f"   ❌ Authentication failed: {e}")
        print("\n   Please use manual authentication as described above.")


if __name__ == "__main__":
    main()

