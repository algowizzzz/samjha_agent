"""
Executor node tests.
"""

import unittest
from unittest.mock import Mock
from external.agent.executor_nodes import (
    investigation_node,
    sql_generation_node,
    safety_validation_node,
    execution_node,
    evaluation_node,
    outcome_node
)
from external.agent.state_types import ExecutorState


class TestExecutorNodes(unittest.TestCase):
    """Test executor nodes."""
    
    def setUp(self):
        """Set up test state."""
        self.tools_registry = Mock()
        self.base_state: ExecutorState = {
            "query_spec": {
                "business_question": "List products",
                "metrics": []
            },
            "query_spec_status": {
                "business_question": {"status": "verified", "source": "user", "notes": "", "blocks_execution": False},
                "start_table_grain": {"status": "verified", "source": "user", "notes": "", "blocks_execution": False},
                "time": {"status": "verified", "source": "domain_md", "notes": "", "blocks_execution": False},
                "metrics": {"status": "verified", "source": "domain_md", "notes": "", "blocks_execution": False}
            },
            "investigation_plan": [],
            "final_sql": None,
            "results": None,
            "executor_report": {},
            "policy_limits": {
                "max_attempts": 3,
                "max_rows": 1000,
                "timeout_seconds": 30,
                "allow_cross_join": False
            },
            "halt_execution": False
        }
    
    def test_investigation_node_skips_when_halted(self):
        """Test investigation node skips when halt_execution is True."""
        state = self.base_state.copy()
        state["halt_execution"] = True
        
        result = investigation_node(state, self.tools_registry)
        # Should return empty dict (no changes)
        self.assertEqual(result, {})
    
    def test_sql_generation_node_gate(self):
        """Test SQL generation node checks gate."""
        state = self.base_state.copy()
        domain_md = "- listing_allows_empty_metrics: true"
        
        # Mock tool
        mock_tool = Mock()
        mock_tool.execute.return_value = {"sql": "SELECT * FROM products LIMIT 10"}
        self.tools_registry.get_tool.return_value = mock_tool
        
        result = sql_generation_node(state, self.tools_registry, domain_md)
        # Should call tool if gate passes
        if not state.get("halt_execution"):
            self.tools_registry.get_tool.assert_called_with("nl_to_sql_planner")
    
    def test_safety_validation_node_blocks_forbidden(self):
        """Test safety validation blocks forbidden operations."""
        state = self.base_state.copy()
        state["final_sql"] = "DELETE FROM products"
        
        mock_tool = Mock()
        mock_tool.execute.return_value = {"allowed": False, "flags": ["FORBIDDEN_OP:DELETE"]}
        self.tools_registry.get_tool.return_value = mock_tool
        
        result = safety_validation_node(state, self.tools_registry)
        # Should set halt_execution if not allowed
        if not mock_tool.execute.return_value["allowed"]:
            self.assertIn("halt_execution", result or {})
    
    def test_outcome_node_builds_error_report(self):
        """Test outcome node builds ERROR report."""
        state = self.base_state.copy()
        state["halt_execution"] = True
        state["final_sql"] = "SELECT * FROM invalid"
        state["last_error"] = "Table not found"
        state["evaluation"] = {"satisfied": False, "issues": ["table_not_found"], "notes": "Error"}
        
        result = outcome_node(state)
        self.assertIn("executor_report", result)
        report = result["executor_report"]
        self.assertEqual(report["status"], "ERROR")
        self.assertIn("error_type", report)
    
    def test_outcome_node_builds_success_report(self):
        """Test outcome node builds SUCCESS report."""
        state = self.base_state.copy()
        state["halt_execution"] = False
        state["final_sql"] = "SELECT * FROM products LIMIT 10"
        state["results"] = {
            "row_count": 10,
            "columns": ["product_id", "name"],
            "rows_preview": []
        }
        state["evaluation"] = {"satisfied": True, "issues": [], "notes": "OK"}
        
        result = outcome_node(state)
        self.assertIn("executor_report", result)
        report = result["executor_report"]
        self.assertEqual(report["status"], "SUCCESS")
        self.assertIn("final_sql", report)


if __name__ == "__main__":
    unittest.main()

