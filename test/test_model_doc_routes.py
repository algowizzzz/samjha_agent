"""
Tests for Model Documentation Agent Flask routes using Flask test client.
This bypasses authentication by using Flask's test client with session_transaction.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask

from external.model_doc.store import ModelDocStore
from external.routes.model_doc_routes import ModelDocRoutes


class DummyAuthManager:
    """Mock auth manager for testing."""
    def validate_session(self, token):
        return {"user": "tester", "user_id": "admin", "user_name": "Admin", "roles": ["admin"]}
    
    def has_tool_access(self, session_data, tool_name):
        return True


class TestModelDocRoutes(unittest.TestCase):
    """Test Model Documentation Agent API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret"
        self.tmpdir = tempfile.TemporaryDirectory()
        
        # Create test directories
        test_data_dir = Path(self.tmpdir.name) / "data" / "model_doc"
        test_data_dir.mkdir(parents=True, exist_ok=True)
        output_dir = test_data_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample codebase
        codebase_dir = test_data_dir / "test_codebase"
        codebase_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test Python files
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
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
''')
        
        (codebase_dir / "__init__.py").write_text('"""Sample codebase."""\n')
        
        self.codebase_path = codebase_dir
        self.store = ModelDocStore(base_dir=Path(self.tmpdir.name) / "state")
        
        # Set up routes
        auth_manager = DummyAuthManager()
        self.routes = ModelDocRoutes(auth_manager, None, None, None)
        self.routes.agent = MagicMock()
        self.routes.agent.build_config.side_effect = (
            lambda overrides=None: {
                "codebase": {"file_extensions": [".py"], "exclude_patterns": ["__pycache__"]},
                "llm": {"model": "claude-3-opus-20240229", "temperature": 0.2},
                "template": {"template_id": "bmo_model_documentation"},
                "output": {"base_dir": str(output_dir), "create_timestamped_dir": True},
            }
        )
        self.routes.store = self.store
        self.routes.register_routes(self.app)
        self.client = self.app.test_client()

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def _login(self):
        """Simulate login by setting session token."""
        with self.client.session_transaction() as session:
            session["token"] = "dummy-token"

    def test_list_templates(self):
        """Test GET /api/model_doc/templates"""
        self._login()
        response = self.client.get("/api/model_doc/templates")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("templates", payload)
        self.assertIsInstance(payload["templates"], list)

    def test_register_codebase(self):
        """Test POST /api/model_doc/documents - Register new codebase"""
        self._login()
        response = self.client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(self.codebase_path),
                "codebase_id": "test_calculator_codebase"
            },
        )
        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        codebase = payload.get("codebase", {})
        self.assertEqual(codebase.get("codebase_id"), "test_calculator_codebase")
        self.assertEqual(codebase.get("status"), "ready")
        self.assertIn("config", codebase.get("state", {}))

    def test_list_codebases(self):
        """Test GET /api/model_doc/documents - List all codebases"""
        self._login()
        
        # First register a codebase
        self.client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(self.codebase_path),
                "codebase_id": "test_list_codebase"
            },
        )
        
        # Then list them
        response = self.client.get("/api/model_doc/documents")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        codebases = payload.get("codebases", [])
        self.assertGreaterEqual(len(codebases), 1)
        
        # Find our codebase
        found = next((c for c in codebases if c.get("codebase_id") == "test_list_codebase"), None)
        self.assertIsNotNone(found)

    def test_get_codebase(self):
        """Test GET /api/model_doc/documents/<codebase_id>"""
        self._login()
        
        # Register codebase
        create_response = self.client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(self.codebase_path),
                "codebase_id": "test_get_codebase"
            },
        )
        codebase_id = create_response.get_json()["codebase"]["codebase_id"]
        
        # Get codebase
        response = self.client.get(f"/api/model_doc/documents/{codebase_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        codebase = payload.get("codebase", {})
        self.assertEqual(codebase.get("codebase_id"), codebase_id)
        self.assertEqual(codebase.get("codebase_path"), str(self.codebase_path))

    def test_get_status(self):
        """Test GET /api/model_doc/documents/<codebase_id>/status"""
        self._login()
        
        # Register codebase
        create_response = self.client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(self.codebase_path),
                "codebase_id": "test_status"
            },
        )
        codebase_id = create_response.get_json()["codebase"]["codebase_id"]
        
        # Get status
        response = self.client.get(f"/api/model_doc/documents/{codebase_id}/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload.get("codebase_id"), codebase_id)
        self.assertIn("status", payload)

    def test_update_config(self):
        """Test PATCH /api/model_doc/documents/<codebase_id>/config"""
        self._login()
        
        # Register codebase
        create_response = self.client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(self.codebase_path),
                "codebase_id": "test_config"
            },
        )
        codebase_id = create_response.get_json()["codebase"]["codebase_id"]
        
        # Update config
        response = self.client.patch(
            f"/api/model_doc/documents/{codebase_id}/config",
            json={
                "config": {
                    "llm": {"temperature": 0.5}
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        codebase = payload.get("codebase", {})
        updated_config = codebase.get("state", {}).get("config", {})
        self.assertEqual(updated_config.get("llm", {}).get("temperature"), 0.5)

    def test_run_phase1(self):
        """Test POST /api/model_doc/documents/<codebase_id>/run_phase1"""
        self._login()
        
        # Register codebase
        create_response = self.client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(self.codebase_path),
                "codebase_id": "test_phase1"
            },
        )
        codebase_id = create_response.get_json()["codebase"]["codebase_id"]
        
        # Mock the agent's phase 1 methods
        with patch.object(self.routes.agent, '_initialise_state') as mock_init, \
             patch.object(self.routes.agent, '_node_list_codebase_files') as mock_list, \
             patch.object(self.routes.agent, '_node_read_code_files') as mock_read, \
             patch.object(self.routes.agent, '_node_parse_code_structure') as mock_parse, \
             patch.object(self.routes.agent, '_node_build_file_hierarchy') as mock_hierarchy, \
             patch.object(self.routes.agent, '_node_compute_code_stats') as mock_stats:
            
            # Set up mock return values
            def mock_state_step(state):
                return state
            
            mock_init.side_effect = lambda state: state
            mock_list.side_effect = mock_state_step
            mock_read.side_effect = mock_state_step
            mock_parse.side_effect = mock_state_step
            mock_hierarchy.side_effect = mock_state_step
            mock_stats.side_effect = lambda state: {
                **state,
                "file_list": [
                    {"file_path": str(self.codebase_path / "calculator.py"), "relative_path": "calculator.py"}
                ],
                "codebase_metadata": {"file_count": 1},
                "code_stats": {"total_lines": 50, "total_classes": 1, "total_functions": 4},
            }
            
            # Run Phase 1
            response = self.client.post(
                f"/api/model_doc/documents/{codebase_id}/run_phase1",
                json={},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            codebase = payload.get("codebase", {})
            self.assertIn("state", codebase)

    @patch('external.routes.model_doc_routes.generate_chat_reply')
    def test_chat(self, mock_chat):
        """Test POST /api/model_doc/chat/<codebase_id>"""
        self._login()
        
        # Register codebase
        create_response = self.client.post(
            "/api/model_doc/documents",
            json={
                "codebase_path": str(self.codebase_path),
                "codebase_id": "test_chat"
            },
        )
        codebase_id = create_response.get_json()["codebase"]["codebase_id"]
        
        # Mock chat response
        mock_chat.return_value = "This codebase contains a Calculator class with arithmetic operations."
        
        # Test chat
        response = self.client.post(
            f"/api/model_doc/chat/{codebase_id}",
            json={"message": "What classes are in this codebase?"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("response", payload)
        self.assertIn("Calculator", payload["response"])

    def test_register_codebase_validation(self):
        """Test that registration validates required fields"""
        self._login()
        
        # Missing codebase_path
        response = self.client.post(
            "/api/model_doc/documents",
            json={"codebase_id": "test"},
        )
        self.assertEqual(response.status_code, 400)
        
        # Invalid path (not a directory)
        response = self.client.post(
            "/api/model_doc/documents",
            json={"codebase_path": str(self.codebase_path / "calculator.py")},
        )
        self.assertEqual(response.status_code, 400)

    def test_get_nonexistent_codebase(self):
        """Test GET with nonexistent codebase_id"""
        self._login()
        response = self.client.get("/api/model_doc/documents/nonexistent")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

