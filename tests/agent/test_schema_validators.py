"""
Schema validation tests.
"""

import unittest
from external.agent.schema_validators import (
    validate_decider_output,
    validate_executor_report,
    validate_query_spec,
    validate_investigation_plan
)


class TestSchemaValidators(unittest.TestCase):
    """Test schema validation functions."""
    
    def test_validate_decider_output_valid(self):
        """Test valid decider_output."""
        valid_output = {
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
                "time": {"column": "", "rule": "no_time", "n_days": None},
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
                "business_question": {"status": "verified", "source": "user", "notes": "", "blocks_execution": False},
                "output_shape": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": False},
                "start_table_grain": {"status": "verified", "source": "user", "notes": "", "blocks_execution": False},
                "time": {"status": "verified", "source": "domain_md", "notes": "", "blocks_execution": False},
                "metrics": {"status": "verified", "source": "domain_md", "notes": "", "blocks_execution": False},
                "dimensions": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": False},
                "filters": {"status": "inferred", "source": "rule", "notes": "", "blocks_execution": False},
                "joins": {"status": "verified", "source": "rule", "notes": "", "blocks_execution": False},
                "aggregation_plan": {"status": "verified", "source": "rule", "notes": "", "blocks_execution": False},
                "validation_checks": {"status": "defaulted", "source": "rule", "notes": "", "blocks_execution": False},
                "performance_guardrails": {"status": "defaulted", "source": "domain_md", "notes": "", "blocks_execution": False}
            },
            "investigation_plan": [],
            "expected_output": "Table of products",
            "stop_conditions": [],
            "ask_user": {"question": "", "why_non_defaultable": "", "what_answer_unblocks": ""},
            "block_reason": ""
        }
        
        valid, error = validate_decider_output(valid_output)
        self.assertTrue(valid, f"Should be valid: {error}")
    
    def test_validate_decider_output_invalid(self):
        """Test invalid decider_output."""
        invalid_output = {
            "action": "INVALID_ACTION",  # Invalid action
            "domain": "ecomm"
        }
        
        valid, error = validate_decider_output(invalid_output)
        self.assertFalse(valid, "Should be invalid")
        self.assertIsNotNone(error)
    
    def test_validate_executor_report_success(self):
        """Test valid SUCCESS executor_report."""
        valid_report = {
            "status": "SUCCESS",
            "final_sql": "SELECT * FROM products LIMIT 10",
            "result_summary": "Returned 10 rows",
            "evaluation": {"satisfied": True, "issues": [], "notes": "OK"},
            "finished_output": "Query executed successfully"
        }
        
        valid, error = validate_executor_report(valid_report)
        self.assertTrue(valid, f"Should be valid: {error}")
    
    def test_validate_executor_report_error(self):
        """Test valid ERROR executor_report."""
        valid_report = {
            "status": "ERROR",
            "error_type": "SQL",
            "failed_checklist_items": ["sql_generation"],
            "what_changed": "SQL syntax error",
            "minimal_fix_suggestion": "Fix column name",
            "last_sql": "SELECT * FROM invalid_table",
            "last_error": "Table not found"
        }
        
        valid, error = validate_executor_report(valid_report)
        self.assertTrue(valid, f"Should be valid: {error}")
    
    def test_validate_query_spec(self):
        """Test query_spec validation."""
        valid_spec = {
            "business_question": "List products",
            "output_shape": {"type": "table", "columns": ["product_id"]},
            "start_table": {"name": "products", "path": "ECommerce/products.csv"},
            "grain": "one row per product",
            "time": {"column": "", "rule": "no_time", "n_days": None},
            "metrics": [],
            "dimensions": [],
            "filters": [],
            "joins": [],
            "aggregation_plan": "No aggregation",
            "validation_checks": [],
            "performance_guardrails": [],
            "defaults_used": [],
            "open_questions": []
        }
        
        valid, error = validate_query_spec(valid_spec)
        self.assertTrue(valid, f"Should be valid: {error}")
    
    def test_validate_investigation_plan(self):
        """Test investigation_plan validation."""
        valid_plan = [
            {
                "step": 1,
                "tool": "inspect_table",
                "args": {"path": "ECommerce/products.csv"},
                "fills_gap": "start_table_grain",
                "success_condition": "Schema verified"
            }
        ]
        
        valid, error = validate_investigation_plan(valid_plan)
        self.assertTrue(valid, f"Should be valid: {error}")


if __name__ == "__main__":
    unittest.main()

