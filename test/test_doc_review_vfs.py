import json
import unittest

from external.doc_review.vfs import DocReviewVFSAdapter


class DocReviewVfsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "structure": {"raw_text": "# Title\nContent"},
            "phase1": {"doc_summary": {"summary": "Test"}},
            "phase2": {
                "chunks": {"Overview": {"text": "Overview section"}},
                "reviews": {"Overview": {"issues": []}},
                "summary_report": {"overall_posture": "ready"},
            },
            "changes": {
                "suggested_changes": [{"id": "CHG-001", "original_text": "Old", "suggested_text": "New"}],
                "applied_change_ids": [],
            },
        }
        self.adapter = DocReviewVFSAdapter(self.state)

    def test_list_root_directories(self) -> None:
        entries = self.adapter.list_dir("/")
        names = {entry["name"] for entry in entries}
        self.assertIn("original", names)
        self.assertIn("phase1", names)
        self.assertIn("phase2", names)

    def test_read_phase1_summary(self) -> None:
        content = self.adapter.read_file("/phase1/doc_summary.json")
        parsed = json.loads(content)
        self.assertEqual(parsed["summary"], "Test")

    def test_write_document_updates_state(self) -> None:
        self.adapter.write_file("/original/document.md", "Updated text")
        self.assertEqual(self.state["structure"]["raw_text"], "Updated text")

    def test_read_section_review(self) -> None:
        entries = self.adapter.list_dir("/phase2/reviews")
        self.assertEqual(entries[0]["name"], "overview.json")
        content = self.adapter.read_file("/phase2/reviews/overview.json")
        self.assertIn("issues", json.loads(content))

    def test_write_suggested_changes_requires_json(self) -> None:
        suggested = [
            {"id": "CHG-100", "original_text": "Old", "suggested_text": "New"},
        ]
        self.adapter.write_file("/changes/suggested_changes.json", json.dumps(suggested))
        self.assertEqual(
            self.state["changes"]["suggested_changes"][0]["id"],
            "CHG-100",
        )

    def test_reject_unknown_write_path(self) -> None:
        with self.assertRaises(PermissionError):
            self.adapter.write_file("/phase1/doc_summary.json", "{}")


if __name__ == "__main__":
    unittest.main()

