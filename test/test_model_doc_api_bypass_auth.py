"""
Test Model Documentation API endpoints BYPASSING authentication.
This script uses Flask's test client with session_transaction() to bypass auth.
This is the standard Flask testing approach - no authentication needed!

Usage:
    python3 test/test_model_doc_api_bypass_auth.py

Based on Reddit/Flask community best practices for testing authenticated routes.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask
    
    # Import route classes
    from external.routes.model_doc_routes import ModelDocRoutes
    from external.model_doc.store import ModelDocStore
    
    FLASK_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project directory with Flask installed.")
    FLASK_AVAILABLE = False
    sys.exit(1)


class DummyAuthManager:
    """Mock auth manager for testing - always allows access."""
    def validate_session(self, token):
        return {
            "user": "tester",
            "user_id": "admin",
            "user_name": "Admin",
            "roles": ["admin"],
            "tools": ["*"]
        }
    
    def has_tool_access(self, session_data, tool_name):
        return True


def create_test_codebase(base_dir: Path) -> Path:
    """Create a sample codebase for testing."""
    codebase_dir = base_dir / "test_codebase"
    codebase_dir.mkdir(parents=True, exist_ok=True)
    
    (codebase_dir / "__init__.py").write_text('"""Test codebase."""\n')
    
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
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
''')
    
    return codebase_dir


def main():
    """Run all tests using Flask test client (authentication bypassed)."""
    print("=" * 70)
    print("Model Documentation API Tests - AUTHENTICATION BYPASSED")
    print("Using Flask Test Client (standard Flask testing approach)")
    print("=" * 70)
    
    # Set up Flask app
    app = Flask(__name__)
    app.secret_key = "test-secret-key"
    app.config["TESTING"] = True  # Enable test mode
    
    # Create temporary directories
    tmpdir = tempfile.TemporaryDirectory()
    test_data_dir = Path(tmpdir.name) / "data" / "model_doc"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    (test_data_dir / "output").mkdir(parents=True, exist_ok=True)
    (test_data_dir / "state").mkdir(parents=True, exist_ok=True)
    
    codebase_path = create_test_codebase(test_data_dir)
    print(f"\n✅ Created test codebase at: {codebase_path}")
    
    # Set up routes with dummy auth (bypasses authentication)
    auth_manager = DummyAuthManager()
    routes = ModelDocRoutes(auth_manager, None, None, None)
    
    # Mock the agent
    routes.agent = MagicMock()
    routes.agent.build_config.side_effect = (
        lambda overrides=None: {
            "codebase": {"file_extensions": [".py"], "exclude_patterns": ["__pycache__"]},
            "llm": {"model": "claude-3-opus-20240229", "temperature": 0.2},
            "template": {"template_id": "bmo_model_documentation"},
            "output": {"base_dir": str(test_data_dir / "output"), "create_timestamped_dir": True},
        }
    )
    
    # Use test store
    routes.store = ModelDocStore(base_dir=test_data_dir / "state")
    routes.register_routes(app)
    
    # Create test client
    client = app.test_client()
    
    # Helper to "login" (sets session token)
    def login():
        """Simulate login by setting session token - bypasses auth checks."""
        with client.session_transaction() as session:
            session["token"] = "test-token"
    
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
            if templates:
                print(f"      Templates: {', '.join([t.get('template_id', '')[:20] for t in templates[:3]])}")
        else:
            print(f"   ❌ Error: {response.data.decode()[:100]}")
        
        # Test 2: Register codebase
        print("\n2️⃣  Testing: POST /api/model_doc/documents")
        login()
        response = client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(codebase_path),
                "codebase_id": "test_calculator"
            },
        )
        results.append(("Register Codebase", response.status_code))
        print(f"   Status: {response.status_code}")
        
        if response.status_code not in [200, 201]:
            print(f"   ❌ Registration failed: {response.data.decode()[:200]}")
            print("\n⚠️  Cannot continue tests without registered codebase.")
            return
        
        codebase_id = response.get_json()["codebase"]["codebase_id"]
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
            print(f"      Path: {codebase.get('codebase_path')}")
        
        # Test 5: Get status
        print(f"\n5️⃣  Testing: GET /api/model_doc/documents/{codebase_id}/status")
        login()
        response = client.get(f"/api/model_doc/documents/{codebase_id}/status")
        results.append(("Get Status", response.status_code))
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.get_json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"      Last node: {data.get('last_node', 'N/A')}")
        
        # Test 6: Update config
        print(f"\n6️⃣  Testing: PATCH /api/model_doc/documents/{codebase_id}/config")
        login()
        response = client.patch(
            f"/api/model_doc/documents/{codebase_id}/config",
            json={"config": {"llm": {"temperature": 0.5}}},
        )
        results.append(("Update Config", response.status_code))
        print(f"   Status: {response.status_code}")
        if response.status_code in [200, 204]:
            print(f"   ✅ Config updated")
        
        # Test 7: Run Phase 1 (with mocked agent methods)
        print(f"\n7️⃣  Testing: POST /api/model_doc/documents/{codebase_id}/run_phase1")
        print("   ⏳ Running Phase 1 workflow...")
        login()
        
        # Mock agent phase 1 methods to return realistic data
        def mock_init(state):
            state.setdefault("file_list", [])
            state.setdefault("file_contents", {})
            return state
        
        def mock_list(state):
            return {
                **state,
                "file_list": [
                    {
                        "file_path": str(codebase_path / "calculator.py"),
                        "relative_path": "calculator.py",
                        "file_size": 300,
                        "line_count": 25,
                    },
                    {
                        "file_path": str(codebase_path / "__init__.py"),
                        "relative_path": "__init__.py",
                        "file_size": 20,
                        "line_count": 1,
                    }
                ]
            }
        
        def mock_read(state):
            file_list = state.get("file_list", [])
            contents = {}
            for file_info in file_list:
                file_path = file_info.get("file_path")
                if Path(file_path).exists():
                    contents[file_path] = Path(file_path).read_text()
            state["file_contents"] = contents
            return state
        
        def mock_parse(state):
            # Add simple AST structure
            for file_info in state.get("file_list", []):
                if file_info.get("file_path", "").endswith("calculator.py"):
                    file_info["ast_structure"] = {
                        "classes": [{"name": "Calculator", "methods": ["add", "subtract"], "line_number": 8}],
                        "functions": [],
                        "imports": [],
                    }
            return state
        
        def mock_hierarchy(state):
            return {
                **state,
                "file_hierarchy": {"type": "directory", "name": "", "children": {}, "files": []},
                "modules": ["calculator"],
                "packages": [],
                "codebase_metadata": {
                    "codebase_path": str(codebase_path),
                    "codebase_id": codebase_id,
                    "file_count": len(state.get("file_list", [])),
                },
            }
        
        def mock_stats(state):
            return {
                **state,
                "code_stats": {
                    "total_lines": 26,
                    "total_classes": 1,
                    "total_functions": 2,
                    "total_methods": 2,
                    "total_imports": 0,
                },
            }
        
        routes.agent._initialise_state = mock_init
        routes.agent._node_list_codebase_files = mock_list
        routes.agent._node_read_code_files = mock_read
        routes.agent._node_parse_code_structure = mock_parse
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
            error_text = response.data.decode()[:200]
            print(f"   ❌ Error: {error_text}")
        
        # Test 8: Chat (mock LLM)
        print(f"\n8️⃣  Testing: POST /api/model_doc/chat/{codebase_id}")
        login()
        with patch('external.routes.model_doc_routes.generate_chat_reply') as mock_chat:
            mock_chat.return_value = "This codebase contains a Calculator class with add and subtract methods."
            response = client.post(
                f"/api/model_doc/chat/{codebase_id}",
                json={"message": "What classes are in this codebase?"},
            )
            results.append(("Chat", response.status_code))
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                response_text = data.get("response", "")
                print(f"   ✅ Chat response received: {len(response_text)} characters")
                print(f"      Preview: {response_text[:60]}...")
            else:
                print(f"   ❌ Error: {response.data.decode()[:100]}")
        
        # Test Summary
        print("\n" + "=" * 70)
        print("Test Results Summary")
        print("=" * 70)
        passed = 0
        failed = 0
        for test_name, status in results:
            if status in [200, 201, 204]:
                status_icon = "✅"
                passed += 1
            else:
                status_icon = "❌"
                failed += 1
            print(f"{status_icon} {test_name}: {status}")
        
        print(f"\n✅ Passed: {passed}/{len(results)}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{len(results)}")
        print("=" * 70)
        print("\n✅ All API endpoint tests completed!")
        print("   (Authentication was bypassed using Flask test client)")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tmpdir.cleanup()


if __name__ == "__main__":
    if not FLASK_AVAILABLE:
        print("❌ Flask not available. Please ensure Flask is installed.")
        sys.exit(1)
    main()

