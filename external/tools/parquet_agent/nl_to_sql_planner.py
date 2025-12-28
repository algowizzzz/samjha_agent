"""
NL to SQL Planner Tool - Rewritten for spec compliance.
Converts completed QuerySpec to SQL.
"""

import json
from tools.base_mcp_tool import BaseMCPTool

try:
    from external.platform.llm import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

from external.platform.prompt_loader import load_prompt


class NLToSQLPlannerTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "nl_to_sql_planner",
            "description": "Convert QuerySpec to SQL",
            "version": "2.0.0",  # New version for spec compliance
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        
        self.llm_client = None
        if LLM_AVAILABLE:
            try:
                self.llm_client = get_llm_client()
                if not self.llm_client.is_available():
                    self.llm_client = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM for SQL planner: {e}")
                self.llm_client = None

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["query_spec", "query_spec_status"],
            "properties": {
                "query_spec": {"type": "object"},
                "query_spec_status": {"type": "object"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["sql"],
            "properties": {
                "sql": {"type": "string"}
            }
        }

    def execute(self, arguments):
        query_spec = arguments["query_spec"]
        query_spec_status = arguments["query_spec_status"]
        
        # Check for blocking gaps
        for field, status in query_spec_status.items():
            if status.get("blocks_execution") and status.get("status") in ["missing", "conflict"]:
                return {"sql": "ERROR: QUERY_SPEC_INCOMPLETE"}
        
        # Fix time.column null issue - convert null to empty string
        if query_spec.get("time", {}).get("column") is None:
            query_spec["time"] = query_spec.get("time", {}).copy()
            query_spec["time"]["column"] = ""
        
        if not self.llm_client:
            raise ValueError("LLM client not available for SQL generation")
        
        # Extract view name from path for SQL generation
        start_table_path = query_spec.get("start_table", {}).get("path", "")
        view_name = None
        if start_table_path:
            # Extract just the filename stem (view name)
            # Handles paths like "ECommerce/sample_sales_data.csv" -> "sample_sales_data"
            from pathlib import Path
            view_name = Path(start_table_path).stem
            
            # Add explicit view name hint to prompt
            view_hint = f"\n\n=== CRITICAL INSTRUCTION ===\nThe table/view name to use in SQL FROM clause is: {view_name}\n\nDO NOT use:\n- {start_table_path}\n- ECommerce.{view_name}\n- {Path(start_table_path).name}\n\nDO use:\n- {view_name}\n\nExample: SELECT * FROM {view_name} LIMIT 5;\n===================\n"
        else:
            view_hint = ""
        
        prompt_template = load_prompt("nl_to_sql_planner")
        prompt = prompt_template.format(
            query_spec=json.dumps(query_spec, indent=2),
            query_spec_status=json.dumps(query_spec_status, indent=2)
        ) + view_hint
        
        response = self.llm_client.invoke_with_prompt(
            system_prompt="",
            user_prompt=prompt,
            response_format=None
        )
        sql = response.strip()
        
        # Clean markdown if present
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        elif sql.startswith("```sql"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        # Post-process SQL to fix any incorrect table references (safety net)
        if view_name and start_table_path:
            import re
            # Fix patterns like "ECommerce.sample_sales_data" or "ECommerce/sample_sales_data"
            # Replace with just the view name
            path_patterns = [
                re.escape(start_table_path),
                re.escape(Path(start_table_path).parent.name + "." + view_name),
                re.escape(Path(start_table_path).parent.name + "/" + view_name),
            ]
            for pattern in path_patterns:
                # Match word boundaries to avoid partial replacements
                sql = re.sub(rf'\b{pattern}\b', view_name, sql, flags=re.IGNORECASE)
        
        return {"sql": sql}

