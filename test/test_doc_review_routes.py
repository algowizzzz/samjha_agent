"""Tests for document review Flask routes."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask

from external.doc_review.store import DocReviewStore
from external.routes.doc_review_routes import DocReviewRoutes


class DummyAuthManager:
    def validate_session(self, token):
        return {"user": "tester"}


class TestDocReviewRoutes(unittest.TestCase):
    """Validate key API endpoints for the doc review workflow."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret"
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = DocReviewStore(base_dir=Path(self.tmpdir.name))
        self.routes = DocReviewRoutes(DummyAuthManager())
        self.routes.agent = MagicMock()
        self.routes.agent.build_config.side_effect = (
            lambda overrides=None: {
                "chunking": {"page_threshold": 50, "default_strategy": "by_h2"},
                "template": {"template_id": "policy_template"},
                "llm": {"model": "claude-3-opus-20240229"},
                "ui": {"append_unmapped_heading": "Additional Sections"},
            }
        )
        self.routes.store = self.store
        self.routes.upload_dir = Path(self.tmpdir.name) / "uploads"
        self.routes.upload_dir.mkdir(parents=True, exist_ok=True)
        self.routes.register_routes(self.app)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _login(self):
        with self.client.session_transaction() as session:
            session["token"] = "dummy-token"

    def test_get_template_detail(self):
        self._login()
        response = self.client.get("/api/doc_review/templates/policy_template")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("content", payload)
        self.assertEqual(payload["template_id"], "policy_template")

    def test_create_document_registers_without_running(self):
        self._login()
        source_path = Path(self.tmpdir.name) / "input.md"
        source_path.write_text("# Sample\n\nBody", encoding="utf-8")

        response = self.client.post(
            "/api/doc_review/documents",
            json={"source_path": str(source_path)},
        )
        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        document = payload.get("document", {})

        self.assertEqual(document.get("status"), "ready")
        self.assertIn("config", document.get("state", {}))
        self.routes.agent.run.assert_not_called()

        stored = self.store.load(document.get("file_id"))
        self.assertEqual(stored["status"], "ready")
        self.assertIn("config", stored.get("state", {}))

    def test_update_markdown_endpoint(self):
        record_state = {
            "file_id": "doc1",
            "improved_md_file_id": "doc1_md",
            "improved_markdown": "Original text",
            "index": {},
            "config": {},
        }
        source_path = str(Path(self.tmpdir.name) / "source.md")
        Path(source_path).write_text("# Source", encoding="utf-8")
        self.store.save("doc1", source_path, record_state, status="completed")

        self._login()
        response = self.client.put(
            "/api/doc_review/documents/doc1/markdown",
            json={"markdown": "# Updated Markdown"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("document", payload)
        updated = self.store.load("doc1")
        self.assertEqual(updated["state"]["improved_markdown"], "# Updated Markdown")

    def test_upload_endpoint(self):
        self._login()
        file_bytes = io.BytesIO(b"# Sample upload")
        response = self.client.post(
            "/api/doc_review/upload",
            data={"file": (file_bytes, "sample.md")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertIn("saved_path", payload)
        self.assertTrue(Path(payload["saved_path"]).exists())

    def test_chat_endpoint_with_selection(self):
        record_state = {
            "file_id": "doc2",
            "improved_markdown": "# Title\nSome content here.",
            "index": {"toc": [{"heading": "Title", "order": 0}]},
            "config": {},
        }
        source_path = str(Path(self.tmpdir.name) / "source2.md")
        Path(source_path).write_text("# Source2", encoding="utf-8")
        self.store.save("doc2", source_path, record_state, status="completed")

        self._login()
        with patch(
            "external.routes.doc_review_routes.generate_chat_reply",
            return_value="Mocked answer",
        ) as mock_chat:
            response = self.client.post(
                "/api/doc_review/chat/doc2",
                json={"message": "Summarise section", "selected_text": "Some content"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["response"], "Mocked answer")
        self.assertEqual(payload["selection"], "Some content")
        mock_chat.assert_called_once()

        updated = self.store.load("doc2")
        history = updated["state"].get("chat_history", [])
        self.assertTrue(history)
        self.assertEqual(history[-1]["selection"], "Some content")

    def test_handle_user_message_endpoint(self):
        record_state = {
            "file_id": "doc3",
            "structure": {"raw_text": "# Title\nBody"},
            "phase1": {},
            "phase2": {},
            "phase3_status": "pending",
        }
        source_path = str(Path(self.tmpdir.name) / "source3.md")
        Path(source_path).write_text("# Source3", encoding="utf-8")
        saved = self.store.save("doc3", source_path, record_state, status="ready")

        self.routes.agent.handle_user_message.return_value = {
            "status": "success",
            "plan": {"summary": "Mock plan", "plan_steps": []},
            "execution_results": {"executed_steps": []},
        }

        self._login()
        response = self.client.post(
            "/api/doc_review/handle_user_message",
            json={"file_id": "doc3", "message": "Run phase 2", "auto_execute": True},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["result"]["status"], "success")
        self.routes.agent.handle_user_message.assert_called_once()
        args, kwargs = self.routes.agent.handle_user_message.call_args
        # First arg is state dict
        self.assertEqual(args[0]["structure"]["raw_text"], "# Title\nBody")
        self.assertEqual(args[1], "Run phase 2")
        self.assertTrue(kwargs.get("auto_execute"))
        self.assertIn("session_id", kwargs)

        updated = self.store.load("doc3")
        self.assertEqual(updated["state"]["structure"]["raw_text"], "# Title\nBody")

    def test_vfs_tree_and_file_endpoints(self):
        record_state = {
            "file_id": "doc4",
            "structure": {"raw_text": "# Title\nBody"},
            "phase1": {"doc_summary": {"summary": "hello"}},
            "phase2": {"chunks": {"Overview": {"text": "Overview text"}}},
            "changes": {"suggested_changes": []},
        }
        source_path = str(Path(self.tmpdir.name) / "source4.md")
        Path(source_path).write_text("# Source4", encoding="utf-8")
        self.store.save("doc4", source_path, record_state, status="ready")

        self._login()
        tree_resp = self.client.get(
            "/api/doc_review/vfs/tree",
            query_string={"file_id": "doc4", "path": "/phase1"},
        )
        self.assertEqual(tree_resp.status_code, 200)
        entries = tree_resp.get_json()["entries"]
        self.assertTrue(any(entry["name"] == "doc_summary.json" for entry in entries))

        file_resp = self.client.get(
            "/api/doc_review/vfs/file",
            query_string={"file_id": "doc4", "path": "/phase1/doc_summary.json"},
        )
        self.assertEqual(file_resp.status_code, 200)
        self.assertIn("hello", file_resp.get_json()["content"])

        patch_resp = self.client.patch(
            "/api/doc_review/vfs/file",
            json={
                "file_id": "doc4",
                "path": "/original/document.md",
                "data": "# Updated\nBody",
            },
        )
        self.assertEqual(patch_resp.status_code, 200)
        updated = self.store.load("doc4")
        self.assertEqual(updated["state"]["structure"]["raw_text"], "# Updated\nBody")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

