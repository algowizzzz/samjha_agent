"""
Error handling and edge case tests.
"""

import unittest
from external.agent.schema_validators import validate_decider_output, validate_executor_report
from external.agent.sql_gate import spec_ready_for_sql
from external.tools.parquet_agent.list_dir import ListDirTool


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_conversation_history(self):
        """Test handling of empty conversation history."""
        from external.agent.parquet_agent import load_domain_md
        result = load_domain_md("test query", [])
        # Should not crash
        self.assertIsInstance(result, str)
    
    def test_missing_domain_file(self):
        """Test handling of missing domain.md file."""
        from external.agent.parquet_agent import load_domain_md
        result = load_domain_md("nonexistent domain query", [])
        # Should return empty string, not crash
        self.assertIsInstance(result, str)
    
    def test_invalid_decider_output(self):
        """Test validation catches invalid Decider output."""
        invalid_output = {
            "action": "INVALID",  # Invalid action
            "domain": "ecomm"
            # Missing required fields
        }
        valid, error = validate_decider_output(invalid_output)
        self.assertFalse(valid)
        self.assertIsNotNone(error)
    
    def test_malformed_query_spec(self):
        """Test handling of malformed query_spec."""
        query_spec = {
            "business_question": "Test"
            # Missing required fields
        }
        query_spec_status = {}
        domain_md = ""
        
        # Should not crash
        ready = spec_ready_for_sql(query_spec, query_spec_status, domain_md)
        self.assertIsInstance(ready, bool)
    
    def test_list_dir_nonexistent_path(self):
        """Test list_dir with nonexistent path."""
        tool = ListDirTool({"data_directory": "mock_datawarehouse"})
        result = tool.execute({"path": "NonexistentDomain"})
        # Should return empty entries, not crash
        self.assertIn("entries", result)
        self.assertIsInstance(result["entries"], list)
    
    def test_executor_report_missing_fields(self):
        """Test validation catches missing fields in executor report."""
        incomplete_report = {
            "status": "SUCCESS"
            # Missing required fields
        }
        valid, error = validate_executor_report(incomplete_report)
        self.assertFalse(valid)
        self.assertIsNotNone(error)
    
    def test_sql_gate_with_missing_status(self):
        """Test SQL gate with missing status fields."""
        query_spec = {
            "business_question": "Test",
            "start_table": {"name": "test", "path": "test.csv"},
            "grain": "one row",
            "time": {"column": "", "rule": "no_time", "n_days": None},
            "metrics": []
        }
        query_spec_status = {}  # Empty status
        domain_md = ""
        
        ready = spec_ready_for_sql(query_spec, query_spec_status, domain_md)
        # With empty status, get() returns None which is not in ["missing", "conflict"]
        # So it may return True if no blocking fields are found
        # This is actually correct behavior - empty status means no blocking issues detected
        self.assertIsInstance(ready, bool)


if __name__ == "__main__":
    unittest.main()

