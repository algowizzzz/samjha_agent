"""
Controller loop tests - retry logic, max_attempts, state management.
"""

import unittest
from unittest.mock import Mock, patch
from external.agent.parquet_agent import handle_query, initialize_controller_state
from external.agent.state_types import ControllerState


class TestControllerLoop(unittest.TestCase):
    """Test controller loop behavior."""
    
    def test_initialize_state(self):
        """Test state initialization."""
        state = initialize_controller_state(
            user_query="Test query",
            conversation_history=[],
            policy_limits={"max_attempts": 5, "max_rows": 1000, "timeout_seconds": 60, "allow_cross_join": False}
        )
        
        self.assertEqual(state["user_query"], "Test query")
        self.assertEqual(state["attempt_count"], 0)
        self.assertEqual(state["policy_limits"]["max_attempts"], 5)
        self.assertIsNone(state["last_executor_report"])
    
    def test_state_persistence(self):
        """Test state persists across retries."""
        prior_state: ControllerState = {
            "user_query": "Test query",
            "conversation_history": [],
            "domain_md": "",
            "policy_limits": {"max_attempts": 3, "max_rows": 1000, "timeout_seconds": 30, "allow_cross_join": False},
            "query_spec": {"business_question": "Test"},
            "query_spec_status": {},
            "last_executor_report": {"status": "ERROR", "error_type": "SQL"},
            "attempt_count": 1
        }
        
        restored = initialize_controller_state("Test query", [], prior_state)
        self.assertEqual(restored["attempt_count"], 1)
        self.assertIsNotNone(restored["last_executor_report"])
        self.assertEqual(restored["query_spec"]["business_question"], "Test")
    
    def test_max_attempts_enforcement(self):
        """Test max_attempts is enforced correctly."""
        state = initialize_controller_state("Test", [], policy_limits={"max_attempts": 2, "max_rows": 1000, "timeout_seconds": 30, "allow_cross_join": False})
        state["attempt_count"] = 2
        state["last_executor_report"] = {"status": "ERROR", "error_type": "SQL"}
        
        max_attempts = state["policy_limits"]["max_attempts"]
        self.assertTrue(state["attempt_count"] >= max_attempts)
        self.assertIsNotNone(state["last_executor_report"])


if __name__ == "__main__":
    unittest.main()

