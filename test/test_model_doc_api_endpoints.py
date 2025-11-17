"""
Test script for Model Documentation Agent API endpoints.
Tests all endpoints using a sample codebase.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
BASE_URL = "http://localhost:5000"
API_BASE = f"{BASE_URL}/api/model_doc"

# Test credentials (adjust if needed)
TEST_USERNAME = "admin"
TEST_PASSWORD = "admin123"  # Default password from config/users.json


def get_auth_session() -> requests.Session:
    """Get authenticated session by logging in."""
    session = requests.Session()
    
    # First, try to get a session cookie by visiting the login page
    try:
        login_page = session.get(f"{BASE_URL}/login", allow_redirects=False)
        # This might set a session cookie
    except:
        pass
    
    # Try web login (form-based)
    login_url = f"{BASE_URL}/login"
    try:
        response = session.post(
            login_url,
            data={"user_id": TEST_USERNAME, "password": TEST_PASSWORD},
            allow_redirects=False,
            headers={"Referer": f"{BASE_URL}/login"}
        )
        
        # If redirected or status is ok, check for session cookie
        if response.status_code in [200, 302] or any('session' in str(c).lower() for c in session.cookies):
            print(f"   ✅ Login successful via web login")
            return session
    except Exception as e:
        print(f"   ⚠️  Web login attempt failed: {e}")
    
    # Fallback to API login
    login_url = f"{BASE_URL}/api/auth/login"
    response = session.post(
        login_url,
        json={"user_id": TEST_USERNAME, "password": TEST_PASSWORD},
        allow_redirects=False,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        token = data.get("token")
        if token:
            # Store token in session cookies for Flask session support
            session.cookies.set("token", token, domain="localhost", path="/")
        print(f"   ✅ Login successful: {data.get('user', {}).get('user_name', TEST_USERNAME)}")
        return session
    
    print(f"   ⚠️  Login failed: {response.status_code}")
    if response.text:
        try:
            error_data = response.json()
            print(f"      Error: {error_data.get('error', response.text)}")
        except:
            print(f"      Response: {response.text[:200]}")
    
    # Try to use environment variable for session cookie
    import os
    flask_session = os.environ.get("FLASK_SESSION_COOKIE")
    if flask_session:
        session.cookies.set("session", flask_session, domain="localhost", path="/")
        print(f"   ℹ️  Using session cookie from FLASK_SESSION_COOKIE environment variable")
        return session
    
    # Return session anyway - will show auth errors but tests can still demonstrate endpoints
    return session


def make_request(method: str, endpoint: str, session: requests.Session = None, **kwargs) -> Dict[str, Any]:
    """Make authenticated API request."""
    url = f"{API_BASE}{endpoint}"
    headers = kwargs.pop("headers", {})
    headers.setdefault("Content-Type", "application/json")
    
    if session:
        response = session.request(method, url, headers=headers, **kwargs)
    else:
        response = requests.request(method, url, headers=headers, **kwargs)
    
    try:
        return {
            "status_code": response.status_code,
            "data": response.json() if response.content else {},
            "headers": dict(response.headers),
        }
    except json.JSONDecodeError:
        return {
            "status_code": response.status_code,
            "data": {"text": response.text},
            "headers": dict(response.headers),
        }


def create_sample_codebase(base_dir: Path) -> Path:
    """Create a sample codebase for testing."""
    codebase_dir = base_dir / "test_sample_codebase"
    codebase_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    (codebase_dir / "__init__.py").write_text('"""Sample codebase for testing."""\n')
    
    # Create main module
    main_module = codebase_dir / "calculator.py"
    main_module.write_text('''"""
Calculator module with basic arithmetic operations.
"""


class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Difference of a and b
        """
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Quotient of a and b
            
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self) -> list:
        """Get calculation history.
        
        Returns:
            List of calculation strings
        """
        return self.history.copy()


def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle.
    
    Args:
        length: Length of the rectangle
        width: Width of the rectangle
        
    Returns:
        Area of the rectangle
    """
    return length * width


def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle.
    
    Args:
        radius: Radius of the circle
        
    Returns:
        Area of the circle
    """
    import math
    return math.pi * radius ** 2
''')
    
    # Create utils module
    utils_dir = codebase_dir / "utils"
    utils_dir.mkdir(exist_ok=True)
    (utils_dir / "__init__.py").write_text('"""Utility functions."""\n')
    
    utils_file = utils_dir / "helpers.py"
    utils_file.write_text('''"""
Helper utility functions.
"""


def format_number(num: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{num:.{decimals}f}"


def validate_input(value: str, min_val: float = None, max_val: float = None) -> float:
    """Validate and convert input string to float.
    
    Args:
        value: Input string
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Converted float value
        
    Raises:
        ValueError: If value is invalid or out of range
    """
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            raise ValueError(f"Value {num} is below minimum {min_val}")
        if max_val is not None and num > max_val:
            raise ValueError(f"Value {num} is above maximum {max_val}")
        return num
    except ValueError as e:
        if "could not convert" in str(e).lower():
            raise ValueError(f"Invalid number: {value}")
        raise
''')
    
    # Create README
    (codebase_dir / "README.md").write_text('''# Sample Codebase

This is a sample codebase for testing the Model Documentation Agent.

## Modules

- `calculator.py`: Calculator class with basic arithmetic operations
- `utils/helpers.py`: Utility helper functions

## Usage

```python
from calculator import Calculator

calc = Calculator()
result = calc.add(5, 3)
print(result)  # 8
```
''')
    
    print(f"✅ Created sample codebase at: {codebase_dir}")
    return codebase_dir


def test_list_templates(session: requests.Session):
    """Test GET /api/model_doc/templates"""
    print("\n🔍 Testing: GET /api/model_doc/templates")
    result = make_request("GET", "/templates", session=session)
    print(f"   Status: {result['status_code']}")
    if result['status_code'] == 200:
        templates = result['data'].get("templates", [])
        print(f"   ✅ Found {len(templates)} templates")
        for template in templates[:3]:  # Show first 3
            print(f"      - {template.get('template_id')}")
    else:
        print(f"   ❌ Error: {result['data']}")
    return result


def test_register_codebase(session: requests.Session, codebase_path: str) -> str:
    """Test POST /api/model_doc/documents"""
    print("\n🔍 Testing: POST /api/model_doc/documents")
    payload = {
        "codebase_path": str(codebase_path),
        "codebase_id": "test_calculator_codebase"
    }
    result = make_request("POST", "/documents", session=session, json=payload)
    print(f"   Status: {result['status_code']}")
    if result['status_code'] in [201, 200]:
        codebase = result['data'].get("codebase", {})
        codebase_id = codebase.get("codebase_id", "test_calculator_codebase")
        print(f"   ✅ Registered codebase: {codebase_id}")
        return codebase_id
    else:
        print(f"   ❌ Error: {result['data']}")
        return None


def test_list_codebases(session: requests.Session):
    """Test GET /api/model_doc/documents"""
    print("\n🔍 Testing: GET /api/model_doc/documents")
    result = make_request("GET", "/documents", session=session)
    print(f"   Status: {result['status_code']}")
    if result['status_code'] == 200:
        codebases = result['data'].get("codebases", [])
        print(f"   ✅ Found {len(codebases)} codebases")
        for codebase in codebases[:3]:  # Show first 3
            print(f"      - {codebase.get('codebase_id')} ({codebase.get('status')})")
    else:
        print(f"   ❌ Error: {result['data']}")
    return result


def test_get_codebase(session: requests.Session, codebase_id: str):
    """Test GET /api/model_doc/documents/<codebase_id>"""
    print(f"\n🔍 Testing: GET /api/model_doc/documents/{codebase_id}")
    result = make_request("GET", f"/documents/{codebase_id}", session=session)
    print(f"   Status: {result['status_code']}")
    if result['status_code'] == 200:
        codebase = result['data'].get("codebase", {})
        print(f"   ✅ Retrieved codebase: {codebase.get('codebase_id')}")
        print(f"      Status: {codebase.get('status')}")
        print(f"      Path: {codebase.get('codebase_path')}")
    else:
        print(f"   ❌ Error: {result['data']}")
    return result


def test_get_status(session: requests.Session, codebase_id: str):
    """Test GET /api/model_doc/documents/<codebase_id>/status"""
    print(f"\n🔍 Testing: GET /api/model_doc/documents/{codebase_id}/status")
    result = make_request("GET", f"/documents/{codebase_id}/status", session=session)
    print(f"   Status: {result['status_code']}")
    if result['status_code'] == 200:
        data = result['data']
        print(f"   ✅ Status: {data.get('status')}")
        print(f"      Last node: {data.get('last_node', 'N/A')}")
    else:
        print(f"   ❌ Error: {result['data']}")
    return result


def test_update_config(session: requests.Session, codebase_id: str):
    """Test PATCH /api/model_doc/documents/<codebase_id>/config"""
    print(f"\n🔍 Testing: PATCH /api/model_doc/documents/{codebase_id}/config")
    payload = {
        "config": {
            "llm": {
                "temperature": 0.3
            }
        }
    }
    result = make_request("PATCH", f"/documents/{codebase_id}/config", session=session, json=payload)
    print(f"   Status: {result['status_code']}")
    if result['status_code'] in [200, 204]:
        print(f"   ✅ Config updated successfully")
    else:
        print(f"   ❌ Error: {result['data']}")
    return result


def test_run_phase1(session: requests.Session, codebase_id: str):
    """Test POST /api/model_doc/documents/<codebase_id>/run_phase1"""
    print(f"\n🔍 Testing: POST /api/model_doc/documents/{codebase_id}/run_phase1")
    print("   ⏳ Running Phase 1 workflow (this may take a moment)...")
    result = make_request("POST", f"/documents/{codebase_id}/run_phase1", session=session, json={})
    print(f"   Status: {result['status_code']}")
    if result['status_code'] == 200:
        codebase = result['data'].get("codebase", {})
        state = codebase.get("state", {})
        file_list = state.get("file_list", [])
        print(f"   ✅ Phase 1 completed")
        print(f"      Files discovered: {len(file_list)}")
        if file_list:
            metadata = state.get("codebase_metadata", {})
            print(f"      Total files: {metadata.get('file_count', 0)}")
            stats = state.get("code_stats", {})
            print(f"      Classes: {stats.get('total_classes', 0)}")
            print(f"      Functions: {stats.get('total_functions', 0)}")
            print(f"      Lines of code: {stats.get('total_lines', 0)}")
    else:
        print(f"   ❌ Error: {result['data']}")
    return result


def test_run_full_workflow(session: requests.Session, codebase_id: str):
    """Test POST /api/model_doc/documents/<codebase_id>/run"""
    print(f"\n🔍 Testing: POST /api/model_doc/documents/{codebase_id}/run")
    print("   ⏳ Running full workflow (this may take several minutes)...")
    result = make_request("POST", f"/documents/{codebase_id}/run", session=session, json={})
    print(f"   Status: {result['status_code']}")
    if result['status_code'] == 200:
        codebase = result['data'].get("codebase", {})
        state = codebase.get("state", {})
        final_doc = state.get("final_documentation")
        print(f"   ✅ Full workflow completed")
        if final_doc:
            print(f"      Documentation generated: {len(final_doc)} characters")
            print(f"      Output path: {state.get('final_documentation_path', 'N/A')}")
        else:
            print(f"      ⚠️  Documentation not generated (may need LLM)")
    else:
        print(f"   ❌ Error: {result['data']}")
    return result


def test_chat(session: requests.Session, codebase_id: str):
    """Test POST /api/model_doc/chat/<codebase_id>"""
    print(f"\n🔍 Testing: POST /api/model_doc/chat/{codebase_id}")
    payload = {
        "message": "What classes are in this codebase?"
    }
    result = make_request("POST", f"/chat/{codebase_id}", session=session, json=payload)
    print(f"   Status: {result['status_code']}")
    if result['status_code'] == 200:
        response = result['data'].get("response", "")
        print(f"   ✅ Chat response received: {len(response)} characters")
        print(f"      Preview: {response[:100]}...")
    else:
        print(f"   ❌ Error: {result['data']}")
        print(f"      Note: This may fail if LLM is not configured")
    return result


def check_server_running() -> bool:
    """Check if the server is running."""
    endpoints_to_try = [
        f"{BASE_URL}/api/health",
        f"{BASE_URL}/api/model_doc/templates",
        f"{BASE_URL}/",
    ]
    
    for endpoint in endpoints_to_try:
        try:
            response = requests.get(endpoint, timeout=2)
            if response.status_code < 500:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue
    
    return False


def main():
    """Run all API endpoint tests."""
    print("=" * 70)
    print("Model Documentation Agent - API Endpoint Tests")
    print("=" * 70)
    
    # Check if server is running
    print(f"\n🔍 Checking if server is running at {BASE_URL}...")
    if not check_server_running():
        print(f"   ❌ Server is not running or not accessible at {BASE_URL}")
        print(f"   Please start the server first:")
        print(f"      python run_server.py")
        print(f"   Or:")
        print(f"      python web/app.py")
        return
    
    print(f"   ✅ Server is running")
    
    # Get authenticated session
    print("\n🔐 Authenticating...")
    session = get_auth_session()
    
    # Check if we have any cookies
    has_cookies = bool(session.cookies)
    if not has_cookies:
        print("\n" + "=" * 70)
        print("⚠️  AUTHENTICATION REQUIRED")
        print("=" * 70)
        print("\nTo test the API endpoints, you need to authenticate first.")
        print("\nOption 1: Login via browser and copy session cookie")
        print(f"   1. Go to: {BASE_URL}/login")
        print("   2. Login with: admin / admin123")
        print("   3. Copy the 'session' cookie from browser DevTools")
        print("   4. Set environment variable:")
        print("      export FLASK_SESSION_COOKIE='<your_session_cookie>'")
        print("   5. Run this script again")
        print("\nOption 2: Use curl with session cookie")
        print("   curl -b 'session=<cookie>' http://localhost:5000/api/model_doc/templates")
        print("\nProceeding with tests (will show authentication errors)...")
        print("=" * 70)
    
    # Create sample codebase
    print("\n📦 Creating sample codebase...")
    test_data_dir = Path(__file__).parent.parent / "data" / "model_doc"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    codebase_path = create_sample_codebase(test_data_dir)
    
    try:
        # Test endpoints
        test_list_templates(session)
        
        codebase_id = test_register_codebase(session, codebase_path)
        if not codebase_id:
            print("\n❌ Failed to register codebase. Stopping tests.")
            return
        
        test_list_codebases(session)
        test_get_codebase(session, codebase_id)
        test_get_status(session, codebase_id)
        test_update_config(session, codebase_id)
        
        # Run Phase 1
        test_run_phase1(session, codebase_id)
        
        # Check status after Phase 1
        print("\n⏳ Waiting 2 seconds before checking status...")
        time.sleep(2)
        test_get_status(session, codebase_id)
        
        # Try chat (may fail if LLM not configured)
        test_chat(session, codebase_id)
        
        # Optionally run full workflow (comment out if you want to skip)
        print("\n" + "=" * 70)
        print("⚠️  Full workflow test skipped by default (takes time)")
        print("   Uncomment test_run_full_workflow() in script to run it")
        print("=" * 70)
        # Uncomment the next line to test full workflow:
        # test_run_full_workflow(session, codebase_id)
        
        print("\n" + "=" * 70)
        print("✅ All endpoint tests completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

