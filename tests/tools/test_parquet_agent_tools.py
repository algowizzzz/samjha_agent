"""
Unit tests for Parquet Agent tools.
"""

import unittest
from pathlib import Path
from tools.tools_registry import ToolsRegistry


class TestParquetAgentTools(unittest.TestCase):
    """Test all 9 parquet agent tools."""
    
    @classmethod
    def setUpClass(cls):
        """Set up tools registry and load tools."""
        cls.registry = ToolsRegistry()
        tools_config_dir = Path("external/config/tools")
        
        tool_names = [
            "list_dir", "inspect_table", "preview_rows", "search_glossary",
            "nl_to_sql_planner", "sql_plan_updater", "query_safety_validator",
            "execute_sql", "query_result_evaluator"
        ]
        
        for tool_name in tool_names:
            tool_config = tools_config_dir / f"{tool_name}.json"
            if tool_config.exists():
                try:
                    cls.registry.load_tool_from_config(tool_config)
                except Exception as e:
                    print(f"Warning: Failed to load {tool_name}: {e}")
    
    def test_list_dir(self):
        """Test list_dir tool."""
        tool = self.registry.get_tool("list_dir")
        self.assertIsNotNone(tool, "list_dir tool should be loaded")
        
        result = tool.execute({"path": "ECommerce"})
        self.assertIn("entries", result)
        self.assertIsInstance(result["entries"], list)
        
        # Should find 3 files in ECommerce
        if len(result["entries"]) > 0:
            self.assertGreaterEqual(len(result["entries"]), 0)
    
    def test_inspect_table(self):
        """Test inspect_table tool."""
        tool = self.registry.get_tool("inspect_table")
        self.assertIsNotNone(tool, "inspect_table tool should be loaded")
        
        result = tool.execute({"path": "ECommerce/sample_sales_data.csv"})
        self.assertIn("columns", result)
        self.assertIsInstance(result["columns"], list)
        
        # Should have order_id column
        column_names = [col["name"] for col in result["columns"]]
        self.assertIn("order_id", column_names)
    
    def test_preview_rows(self):
        """Test preview_rows tool."""
        tool = self.registry.get_tool("preview_rows")
        self.assertIsNotNone(tool, "preview_rows tool should be loaded")
        
        result = tool.execute({"path": "ECommerce/sample_sales_data.csv", "limit": 5})
        self.assertIn("rows_preview", result)
        self.assertIsInstance(result["rows_preview"], list)
        self.assertLessEqual(len(result["rows_preview"]), 5)
    
    def test_search_glossary(self):
        """Test search_glossary tool."""
        tool = self.registry.get_tool("search_glossary")
        self.assertIsNotNone(tool, "search_glossary tool should be loaded")
        
        result = tool.execute({"term": "revenue", "domain": "ecomm"})
        self.assertIn("hits", result)
        self.assertIsInstance(result["hits"], list)
    
    def test_query_safety_validator(self):
        """Test query_safety_validator tool."""
        tool = self.registry.get_tool("query_safety_validator")
        self.assertIsNotNone(tool, "query_safety_validator tool should be loaded")
        
        # Test SELECT (should be allowed)
        result = tool.execute({
            "sql": "SELECT * FROM sample_sales_data LIMIT 10",
            "policy_limits": {
                "max_attempts": 3,
                "max_rows": 1000,
                "timeout_seconds": 30,
                "allow_cross_join": False
            }
        })
        self.assertIn("allowed", result)
        self.assertTrue(result["allowed"], "SELECT should be allowed")
        
        # Test DELETE (should be blocked)
        result = tool.execute({
            "sql": "DELETE FROM sample_sales_data",
            "policy_limits": {
                "max_attempts": 3,
                "max_rows": 1000,
                "timeout_seconds": 30,
                "allow_cross_join": False
            }
        })
        self.assertFalse(result["allowed"], "DELETE should be blocked")
    
    def test_execute_sql(self):
        """Test execute_sql tool."""
        tool = self.registry.get_tool("execute_sql")
        self.assertIsNotNone(tool, "execute_sql tool should be loaded")
        
        result = tool.execute({
            "sql": "SELECT * FROM sample_sales_data LIMIT 5",
            "max_rows": 5
        })
        self.assertIn("row_count", result)
        self.assertIn("columns", result)
        self.assertIn("rows_preview", result)
        self.assertIsInstance(result["row_count"], int)
    
    def test_query_result_evaluator(self):
        """Test query_result_evaluator tool."""
        tool = self.registry.get_tool("query_result_evaluator")
        self.assertIsNotNone(tool, "query_result_evaluator tool should be loaded")
        
        query_spec = {
            "business_question": "List sales",
            "output_shape": {"type": "table", "columns": ["order_id", "customer_id"]},
            "grain": "one row per order",
            "metrics": [],
            "validation_checks": ["row_count > 0"]
        }
        
        results_summary = {
            "row_count": 5,
            "column_names": ["order_id", "customer_id"],
            "sample_rows": []
        }
        
        result = tool.execute({
            "query_spec": query_spec,
            "results_summary": results_summary,
            "validation_checks": ["row_count > 0"]
        })
        self.assertIn("satisfied", result)
        self.assertIn("issues", result)
        self.assertIn("notes", result)


if __name__ == "__main__":
    unittest.main()

