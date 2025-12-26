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


NL_TO_SQL_PROMPT = """
ROLE
You are the SQL Generator.
Your job is to convert a completed Query Spec into SQL.

HARD CONSTRAINTS
- Output SQL ONLY. No prose, no markdown.
- You MUST NOT invent tables, columns, joins, metrics, or filters.
- You MUST NOT guess missing fields.
- You MUST NOT modify the Query Spec.
- If the Query Spec is incomplete for SQL generation, output exactly:
  ERROR: QUERY_SPEC_INCOMPLETE

INPUTS
Query Spec:
{query_spec}

Query Spec Status:
{query_spec_status}

SQL RULES
- **CRITICAL TABLE NAMING**: Use ONLY the view name (file stem), NOT the full path.
  - If query_spec.start_table.path is "ECommerce/sample_sales_data.csv", use view name: "sample_sales_data"
  - If query_spec.start_table.path is "ECommerce/sample_customer_data.csv", use view name: "sample_customer_data"
  - Views are registered by file stem name only (without path prefix).
  - Example: FROM sample_sales_data (CORRECT), NOT FROM ECommerce.sample_sales_data (WRONG)

- Apply query_spec.filters exactly.
- Implement query_spec.aggregation_plan and preserve query_spec.grain.
- Apply time filtering based on query_spec.time:
  - If time.rule = "no_time": No time filter
  - If time.rule = "last_n_days": WHERE column >= CURRENT_DATE - INTERVAL 'n' DAY
  - If time.rule = "date_range": WHERE column BETWEEN 'start' AND 'end'
  - Use DuckDB syntax, NOT MySQL DATE_SUB()
- Respect query_spec.performance_guardrails.

OUTPUT
Return a single SQL string.
"""


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
        
        prompt = NL_TO_SQL_PROMPT.format(
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

