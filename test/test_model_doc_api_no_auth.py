"""
Test script for Model Documentation Agent API endpoints WITHOUT authentication.
Uses direct route calls or test client approach to bypass authentication.
Run this with the Flask app in test mode, or modify routes temporarily.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import Flask components for test client
try:
    from flask import Flask
    from external.routes.model_doc_routes import ModelDocRoutes
    from external.model_doc.store import ModelDocStore
    from unittest.mock import MagicMock
    
    class DummyAuthManager:
        """Mock auth manager that always allows access."""
        def validate_session(self, token):
            return {"user": "tester", "user_id": "admin", "user_name": "Admin", "roles": ["admin"]}
        
        def has_tool_access(self, session_data, tool_name):
            return True
    
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask not available. Will use HTTP testing instead.")
    import requests


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
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b.
        
        Raises:
            ValueError: If b is zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
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


def test_with_flask_client():
    """Test endpoints using Flask test client (bypasses authentication)."""
    print("\n" + "=" * 70)
    print("Testing with Flask Test Client (Authentication Bypassed)")
    print("=" * 70)
    
    # Set up Flask app
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.config["TESTING"] = True
    
    tmpdir = tempfile.TemporaryDirectory()
    test_data_dir = Path(tmpdir.name) / "data" / "model_doc"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    codebase_path = create_sample_codebase(test_data_dir)
    
    # Set up routes with dummy auth
    auth_manager = DummyAuthManager()
    routes = ModelDocRoutes(auth_manager, None, None, None)
    routes.agent = MagicMock()
    routes.agent.build_config.side_effect = (
        lambda overrides=None: {
            "codebase": {"file_extensions": [".py"], "exclude_patterns": ["__pycache__"]},
            "llm": {"model": "claude-3-opus-20240229", "temperature": 0.2},
            "template": {"template_id": "bmo_model_documentation"},
            "output": {"base_dir": str(test_data_dir / "output"), "create_timestamped_dir": True},
        }
    )
    routes.store = ModelDocStore(base_dir=Path(tmpdir.name) / "state")
    routes.register_routes(app)
    
    client = app.test_client()
    
    def login():
        """Simulate login."""
        with client.session_transaction() as session:
            session["token"] = "dummy-token"
    
    results = []
    
    try:
        # Test 1: List templates
        print("\n1️⃣  Testing: GET /api/model_doc/templates")
        login()
        response = client.get("/api/model_doc/templates")
        results.append(("List Templates", response.status_code))
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            templates = data.get("templates", [])
            print(f"   ✅ Found {len(templates)} templates")
        else:
            print(f"   ❌ Error: {response.data.decode()[:100]}")
        
        # Test 2: Register codebase
        print("\n2️⃣  Testing: POST /api/model_doc/documents")
        login()
        response = client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(codebase_path),
                "codebase_id": "test_calculator_codebase"
            },
        )
        results.append(("Register Codebase", response.status_code))
        print(f"   Status: {response.status_code}")
        if response.status_code in [200, 201]:
            data = response.get_json()
            codebase_id = data.get("codebase", {}).get("codebase_id", "test_calculator_codebase")
            print(f"   ✅ Registered codebase: {codebase_id}")
            
            # Test 3: List codebases
            print("\n3️⃣  Testing: GET /api/model_doc/documents")
            login()
            response = client.get("/api/model_doc/documents")
            results.append(("List Codebases", response.status_code))
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                codebases = data.get("codebases", [])
                print(f"   ✅ Found {len(codebases)} codebases")
            
            # Test 4: Get codebase
            print(f"\n4️⃣  Testing: GET /api/model_doc/documents/{codebase_id}")
            login()
            response = client.get(f"/api/model_doc/documents/{codebase_id}")
            results.append(("Get Codebase", response.status_code))
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                codebase = data.get("codebase", {})
                print(f"   ✅ Retrieved: {codebase.get('codebase_id')}")
                print(f"      Status: {codebase.get('status')}")
            
            # Test 5: Get status
            print(f"\n5️⃣  Testing: GET /api/model_doc/documents/{codebase_id}/status")
            login()
            response = client.get(f"/api/model_doc/documents/{codebase_id}/status")
            results.append(("Get Status", response.status_code))
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"   ✅ Status: {data.get('status')}")
            
            # Test 6: Update config
            print(f"\n6️⃣  Testing: PATCH /api/model_doc/documents/{codebase_id}/config")
            login()
            response = client.patch(
                f"/api/model_doc/documents/{codebase_id}/config",
                json={"config": {"llm": {"temperature": 0.3}}},
            )
            results.append(("Update Config", response.status_code))
            print(f"   Status: {response.status_code}")
            if response.status_code in [200, 204]:
                print(f"   ✅ Config updated")
            
            # Test 7: Run Phase 1 (with mocked agent methods)
            print(f"\n7️⃣  Testing: POST /api/model_doc/documents/{codebase_id}/run_phase1")
            print("   ⏳ Running Phase 1 workflow...")
            login()
            
            # Mock agent phase 1 methods
            def mock_init_state(state):
                return {**state, "file_list": []}
            
            def mock_list_files(state):
                return {
                    **state,
                    "file_list": [
                        {
                            "file_path": str(codebase_path / "calculator.py"),
                            "relative_path": "calculator.py",
                            "file_size": 500,
                            "line_count": 40,
                        }
                    ]
                }
            
            def mock_read_files(state):
                file_list = state.get("file_list", [])
                state["file_contents"] = {}
                for file_info in file_list:
                    file_path = file_info.get("file_path")
                    if Path(file_path).exists():
                        state["file_contents"][file_path] = Path(file_path).read_text()
                return state
            
            def mock_parse_ast(state):
                return state
            
            def mock_hierarchy(state):
                return {
                    **state,
                    "file_hierarchy": {},
                    "modules": [],
                    "packages": [],
                    "codebase_metadata": {"file_count": len(state.get("file_list", []))},
                }
            
            def mock_stats(state):
                return {
                    **state,
                    "code_stats": {
                        "total_lines": 40,
                        "total_classes": 1,
                        "total_functions": 4,
                        "total_methods": 4,
                    },
                }
            
            routes.agent._initialise_state = mock_init_state
            routes.agent._node_list_codebase_files = mock_list_files
            routes.agent._node_read_code_files = mock_read_files
            routes.agent._node_parse_code_structure = mock_parse_ast
            routes.agent._node_build_file_hierarchy = mock_hierarchy
            routes.agent._node_compute_code_stats = mock_stats
            
            response = client.post(
                f"/api/model_doc/documents/{codebase_id}/run_phase1",
                json={},
            )
            results.append(("Run Phase 1", response.status_code))
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
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
                print(f"   ❌ Error: {response.data.decode()[:200]}")
            
            # Test 8: Chat
            print(f"\n8️⃣  Testing: POST /api/model_doc/chat/{codebase_id}")
            login()
            try:
                from unittest.mock import patch
                with patch('external.routes.model_doc_routes.generate_chat_reply', return_value="This codebase contains a Calculator class."):
                    response = client.post(
                        f"/api/model_doc/chat/{codebase_id}",
                        json={"message": "What classes are in this codebase?"},
                    )
                    results.append(("Chat", response.status_code))
                    print(f"   Status: {response.status_code}")
                    if response.status_code == 200:
                        data = response.get_json()
                        print(f"   ✅ Chat response received: {len(data.get('response', ''))} characters")
                    else:
                        print(f"   ⚠️  Chat failed (may need LLM): {response.data.decode()[:100]}")
            except Exception as e:
                print(f"   ⚠️  Chat test skipped: {e}")
                results.append(("Chat", "SKIPPED"))
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tmpdir.cleanup()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    for test_name, status in results:
        status_icon = "✅" if status in [200, 201, 204] else "❌"
        print(f"{status_icon} {test_name}: {status}")
    
    passed = sum(1 for _, s in results if s in [200, 201, 204])
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    print("=" * 70)


def main():
    """Main entry point."""
    print("=" * 70)
    print("Model Documentation Agent - API Endpoint Tests (No Auth)")
    print("=" * 70)
    
    if FLASK_AVAILABLE:
        print("\n✅ Using Flask Test Client (authentication bypassed)")
        test_with_flask_client()
    else:
        print("\n❌ Flask not available. Please install Flask:")
        print("   pip install flask")
        print("\nOr use the HTTP test script with manual authentication:")
        print("   python3 test/test_model_doc_api_endpoints.py")


if __name__ == "__main__":
    main()

