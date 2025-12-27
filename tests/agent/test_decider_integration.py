"""
Decider-specific integration tests.
"""

import unittest
from unittest.mock import Mock, patch
from external.agent.decider import run_decider
from external.agent.state_types import ControllerState


class TestDeciderIntegration(unittest.TestCase):
    """Test Decider component with mocked LLM."""
    
    def setUp(self):
        """Set up test state."""
        self.base_state: ControllerState = {
            "user_query": "List products",
            "conversation_history": [],
            "domain_md": """
# Domain: ecomm
## 3) Listing rules
- listing_allows_empty_metrics: true
- listing_default_limit: 50
""",
            "policy_limits": {
                "max_attempts": 3,
                "max_rows": 5000,
                "timeout_seconds": 30,
                "allow_cross_join": False
            },
            "query_spec": {},
            "query_spec_status": {},
            "last_executor_report": None,
            "attempt_count": 0
        }
    
    @patch('external.agent.decider.get_llm_client')
    def test_decider_execute_action(self, mock_llm_client):
        """Test Decider produces EXECUTE action."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.invoke_with_prompt.return_value = """{
  "action": "EXECUTE",
  "domain": "ecomm",
  "intent": "NEW_QUERY",
  "decisions": {
    "comprehension": "INTELLIGIBLE",
    "determinacy": "DETERMINED",
    "clarification_need": "DEFAULT_OK"
  },
  "query_spec": {
    "business_question": "List products",
    "output_shape": {"type": "table", "columns": ["product_id"]},
    "start_table": {"name": "products", "path": "ECommerce/sample_inventory_data.csv"},
    "grain": "one row per product",
    "time": {"column": "", "rule": "no_time", "n_days": null},
    "metrics": [],
    "dimensions": [],
    "filters": [],
    "joins": [],
    "aggregation_plan": "No aggregation",
    "validation_checks": [],
    "performance_guardrails": [],
    "defaults_used": [],
    "open_questions": []
  },
  "query_spec_status": {
    "business_question": {"status": "verified", "source": "user", "notes": "", "blocks_execution": false},
    "output_shape": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": false},
    "start_table_grain": {"status": "verified", "source": "user", "notes": "", "blocks_execution": false},
    "time": {"status": "verified", "source": "domain_md", "notes": "", "blocks_execution": false},
    "metrics": {"status": "verified", "source": "domain_md", "notes": "", "blocks_execution": false},
    "dimensions": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": false},
    "filters": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": false},
    "joins": {"status": "verified", "source": "rule", "notes": "", "blocks_execution": false},
    "aggregation_plan": {"status": "verified", "source": "rule", "notes": "", "blocks_execution": false},
    "validation_checks": {"status": "defaulted", "source": "rule", "notes": "", "blocks_execution": false},
    "performance_guardrails": {"status": "defaulted", "source": "domain_md", "notes": "", "blocks_execution": false}
  },
  "investigation_plan": [],
  "expected_output": "Table of products",
  "stop_conditions": [],
  "ask_user": {"question": "", "why_non_defaultable": "", "what_answer_unblocks": ""},
  "block_reason": ""
}"""
        mock_llm_client.return_value = mock_client
        
        result = run_decider(self.base_state)
        self.assertEqual(result["action"], "EXECUTE")
        self.assertIn("query_spec", result)
        self.assertIn("query_spec_status", result)
    
    @patch('external.agent.decider.get_llm_client')
    def test_decider_ask_user_action(self, mock_llm_client):
        """Test Decider produces ASK_USER action."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.invoke_with_prompt.return_value = """{
  "action": "ASK_USER",
  "domain": "ecomm",
  "intent": "NEW_QUERY",
  "decisions": {
    "comprehension": "INTELLIGIBLE",
    "determinacy": "UNDERDETERMINED",
    "clarification_need": "ASK_REQUIRED"
  },
  "query_spec": {
    "business_question": "Get revenue",
    "output_shape": {"type": "table", "columns": ["revenue"]},
    "start_table": {"name": "", "path": ""},
    "grain": "",
    "time": {"column": "order_date", "rule": "last_n_days", "n_days": 30},
    "metrics": ["revenue"],
    "dimensions": [],
    "filters": [],
    "joins": [],
    "aggregation_plan": "Aggregate revenue",
    "validation_checks": [],
    "performance_guardrails": [],
    "defaults_used": [],
    "open_questions": ["Which entity scope?"]
  },
  "query_spec_status": {
    "business_question": {"status": "verified", "source": "user", "notes": "", "blocks_execution": false},
    "output_shape": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": false},
    "start_table_grain": {"status": "missing", "source": "rule", "notes": "Need to know scope", "blocks_execution": true},
    "time": {"status": "defaulted", "source": "domain_md", "notes": "", "blocks_execution": true},
    "metrics": {"status": "verified", "source": "user", "notes": "", "blocks_execution": true},
    "dimensions": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": false},
    "filters": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": false},
    "joins": {"status": "missing", "source": "rule", "notes": "", "blocks_execution": true},
    "aggregation_plan": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": true},
    "validation_checks": {"status": "defaulted", "source": "rule", "notes": "", "blocks_execution": false},
    "performance_guardrails": {"status": "defaulted", "source": "domain_md", "notes": "", "blocks_execution": false}
  },
  "investigation_plan": [],
  "expected_output": "Revenue value",
  "stop_conditions": [],
  "ask_user": {
    "question": "When you say revenue, do you mean payments received or order totals?",
    "why_non_defaultable": "Different definitions change the result",
    "what_answer_unblocks": "Selecting the revenue definition determines the correct start table"
  },
  "block_reason": ""
}"""
        mock_llm_client.return_value = mock_client
        
        result = run_decider(self.base_state)
        self.assertEqual(result["action"], "ASK_USER")
        self.assertIn("ask_user", result)
        self.assertNotEqual(result["ask_user"]["question"], "")


if __name__ == "__main__":
    unittest.main()

