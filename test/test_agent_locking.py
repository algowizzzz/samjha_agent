import json
import tempfile
import unittest
from pathlib import Path

from external.agent.doc_review_agent import DocReviewAgent


class AgentLockingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DocReviewAgent()
        self.state = self.agent._initialise_state(
            source=Path(__file__),
            run_id="testrun-locking",
            template_id="policy_template",
        )

    def test_acquire_and_release_lock(self) -> None:
        session = "session-123"
        self.agent._acquire_run_lock(self.state, session)
        self.assertEqual(self.state["locked_by"], session)

        # Re-acquiring with same session is a no-op
        self.agent._acquire_run_lock(self.state, session)

        with self.assertRaises(RuntimeError):
            self.agent._acquire_run_lock(self.state, "session-456")

        self.agent._release_run_lock(self.state, session)
        self.assertIsNone(self.state["locked_by"])

    def test_agent_transcript_log_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.agent.agent_log_root = Path(tmpdir)
            entry = {"user_message": "Run full review", "status": "success"}
            self.agent._append_agent_transcript_log(self.state, entry)

            log_path = Path(tmpdir) / self.state["run_id"] / "agent_transcript.jsonl"
            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8").strip()
            self.assertTrue(content)
            logged = json.loads(content)
            self.assertEqual(logged["user_message"], "Run full review")


if __name__ == "__main__":
    unittest.main()

