"""
Contract compliance tests - ensure all outputs match schemas.
"""

import unittest
from external.agent.schema_validators import (
    validate_decider_output,
    validate_executor_report,
    validate_query_spec,
    validate_query_spec_status,
    validate_investigation_plan,
    validate_policy_limits
)


class TestContractCompliance(unittest.TestCase):
    """Test that all contracts match their schemas."""
    
    def test_decider_output_compliance(self):
        """Test Decider output matches schema."""
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
        self.assertTrue(valid, f"Decider output should be valid: {error}")
    
    def test_executor_report_success_compliance(self):
        """Test SUCCESS executor report matches schema."""
        report = {
            "status": "SUCCESS",
            "final_sql": "SELECT * FROM products LIMIT 10",
            "result_summary": "Returned 10 rows",
            "evaluation": {"satisfied": True, "issues": [], "notes": "OK"},
            "finished_output": "Query executed successfully"
        }
        
        valid, error = validate_executor_report(report)
        self.assertTrue(valid, f"SUCCESS report should be valid: {error}")
    
    def test_executor_report_error_compliance(self):
        """Test ERROR executor report matches schema."""
        report = {
            "status": "ERROR",
            "error_type": "SQL",
            "failed_checklist_items": ["sql_generation"],
            "what_changed": "SQL syntax error",
            "minimal_fix_suggestion": "Fix column name",
            "last_sql": "SELECT * FROM invalid_table",
            "last_error": "Table not found"
        }
        
        valid, error = validate_executor_report(report)
        self.assertTrue(valid, f"ERROR report should be valid: {error}")
    
    def test_query_spec_compliance(self):
        """Test QuerySpec matches schema."""
        spec = {
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
        
        valid, error = validate_query_spec(spec)
        self.assertTrue(valid, f"QuerySpec should be valid: {error}")
    
    def test_query_spec_status_compliance(self):
        """Test QuerySpecStatus matches schema."""
        status = {
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
        }
        
        valid, error = validate_query_spec_status(status)
        self.assertTrue(valid, f"QuerySpecStatus should be valid: {error}")
    
    def test_investigation_plan_compliance(self):
        """Test InvestigationPlan matches schema."""
        plan = [
            {
                "step": 1,
                "tool": "inspect_table",
                "args": {"path": "ECommerce/products.csv"},
                "fills_gap": "start_table_grain",
                "success_condition": "Schema verified"
            }
        ]
        
        valid, error = validate_investigation_plan(plan)
        self.assertTrue(valid, f"InvestigationPlan should be valid: {error}")
    
    def test_policy_limits_compliance(self):
        """Test PolicyLimits matches schema."""
        limits = {
            "max_attempts": 3,
            "max_rows": 5000,
            "timeout_seconds": 30,
            "allow_cross_join": False
        }
        
        valid, error = validate_policy_limits(limits)
        self.assertTrue(valid, f"PolicyLimits should be valid: {error}")


if __name__ == "__main__":
    unittest.main()

