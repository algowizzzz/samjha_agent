"""
NL to SQL Planner Tool
Converts a natural language query and schema hints into a structured SQL plan.
Uses LLM when available, falls back to heuristics.
"""
from typing import Dict, Any, List
from datetime import datetime
from tools.base_mcp_tool import BaseMCPTool
import json

try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None

try:
    from agent.llm_client import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class NLToSQLPlannerTool(BaseMCPTool):
    def __init__(self, config: Dict = None):
        default_config = {
            "name": "nl_to_sql_planner",
            "description": "Convert natural language query to SQL with confidence score and explanation",
            "version": "1.0.0",
            "enabled": True,
            "use_llm": True,  # Set to False to force heuristics
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.preview_rows = self.config.get("preview_rows", 100)
        self.data_directory = self.config.get("data_directory", "data/duckdb")
        self.db_path = f"{self.data_directory}/duckdb_analytics.db"
        self.use_llm = self.config.get("use_llm", True) and LLM_AVAILABLE
        
        # Initialize LLM if available
        self.llm_client = None
        if self.use_llm:
            try:
                self.llm_client = get_llm_client()
                if not self.llm_client.is_available():
                    print("⚠ LLM not available, using heuristics for NL-to-SQL")
                    self.use_llm = False
            except Exception as e:
                print(f"⚠ Failed to initialize LLM for planner: {e}")
                self.use_llm = False

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "table_schema": {"type": "object", "description": "Table schema metadata"},
                "docs_meta": {"type": "array", "description": "Optional doc metadata"},
                "previous_clarifications": {"type": "array", "description": "Past clarifications"},
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "plan_quality": {"type": "string"},
                "plan_explain": {"type": "string"},
                "clarification_questions": {"type": "array"},
                "timestamp": {"type": "string"},
            },
            "required": ["plan", "plan_quality", "plan_explain"],
        }

    def _list_tables(self) -> List[str]:
        if duckdb is None:
            return []
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            # Get both tables and views
            tables = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()]
            views = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.views WHERE table_schema = 'main'").fetchall()]
            conn.close()
            all_tables = list(set(tables + views))
            print(f"[NLToSQLPlanner] Found tables/views: {all_tables}")  # Debug
            return all_tables
        except Exception as e:
            print(f"[NLToSQLPlanner] Error listing tables: {e}")  # Debug
            return []

    def _simple_guess(self, nl_query: str, available_tables: List[str]) -> Dict[str, Any]:
        """
        Improved heuristic with better keyword matching
        """
        text = nl_query.lower()
        chosen = None
        
        # Keyword matching for common queries
        if "sales" in text or "revenue" in text or "sold" in text:
            for t in available_tables:
                if "sales" in t.lower():
                    chosen = t
                    break
        elif "customer" in text or "client" in text:
            for t in available_tables:
                if "customer" in t.lower():
                    chosen = t
                    break
        elif "inventory" in text or "stock" in text or "product" in text:
            for t in available_tables:
                if "inventory" in t.lower():
                    chosen = t
                    break
        
        # Fallback: check if table name is mentioned
        if not chosen:
            for t in available_tables:
                if t.lower() in text:
                    chosen = t
                    break
        
        # Last resort: use first available table
        if not chosen and available_tables:
            chosen = available_tables[0]
        
        # Build SQL with LIMIT
        if chosen:
            if "top" in text or "limit" in text:
                # Extract number if mentioned
                import re
                numbers = re.findall(r'\d+', text)
                limit = int(numbers[0]) if numbers else 10
                sql = f"SELECT * FROM {chosen} LIMIT {limit}"
            elif "total" in text or "sum" in text:
                sql = f"SELECT SUM(*) as total FROM {chosen} LIMIT 1"
            else:
                sql = f"SELECT * FROM {chosen} LIMIT {min(self.preview_rows, 10)}"
        else:
            sql = "SELECT 1"
        
        return {
            "type": "sql_plan",
            "sql": sql,
            "limits": {"preview_rows": self.preview_rows},
            "target_table": chosen,
        }

    def _llm_plan(self, nl_query: str, table_schema: Dict[str, Any], available_tables: List[str], docs_meta: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to generate SQL plan"""
        # Get prompts from config if available, otherwise use defaults
        from agent.config import QueryAgentConfig
        cfg = QueryAgentConfig()
        
        system_prompt = cfg.get_nested("prompts", "planner_system", default="""You are a DuckDB SQL query generator. Convert natural language to valid DuckDB SQL.

CRITICAL RULES:
1. Use PostgreSQL/DuckDB syntax: SELECT columns FROM table ORDER BY col LIMIT N
2. NEVER use "SELECT TOP N" - that's SQL Server syntax and will cause errors
3. Use exact table names from the schema (e.g., "sample_sales_data")
4. Use "SELECT *" when columns are unknown
5. Always add "LIMIT 10" at the end

Example correct query:
SELECT * FROM sample_sales_data ORDER BY amount DESC LIMIT 5

Example WRONG query (do not generate):
SELECT TOP 5 * FROM sample_sales_data

Respond with JSON:
{
  "sql": "SELECT * FROM table_name ORDER BY column DESC LIMIT 10",
  "plan_quality": "high|medium|low",
  "plan_explanation": "Brief explanation",
  "clarification_questions": [],
  "target_table": "table_name"
}""")

        # Build schema summary with business context
        schema_summary = "Available tables/views:\n"
        for table_name, schema_info in table_schema.items():
            cols = schema_info.get('columns', [])
            if cols:
                col_str = ', '.join([f"{c['name']}" for c in cols[:15]])  # Show column names
                schema_summary += f"\n{table_name}:\n"
                schema_summary += f"  Columns: {col_str}\n"
            else:
                schema_summary += f"- {table_name}\n"
        
        if not table_schema:
            schema_summary = f"Available tables/views: {', '.join(available_tables)}\n"
        
        # Add business context from docs_meta
        business_context = ""
        if docs_meta:
            for doc in docs_meta:
                if doc.get("type") == "business_glossary":
                    business_context += "\nBusiness Glossary (important!):\n"
                    for term, definition in doc.get("glossary", {}).items():
                        business_context += f"  - {term}: {definition}\n"
                elif "table" in doc:
                    business_context += f"\n{doc['table']} - {doc.get('business_context', '')}\n"
        
        # Load procedural knowledge from data_dictionary
        procedural_knowledge = ""
        try:
            import os
            data_dict_path = "config/data_dictionary.json"
            if os.path.exists(data_dict_path):
                with open(data_dict_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    procedural_knowledge = data_dict.get("procedural_knowledge", "")
        except Exception:
            pass  # Optional, graceful failure
        
        print(f"[NLToSQLPlanner] Schema summary for LLM:\n{schema_summary}")  # Debug

        # Get user prompt template from config
        user_template = cfg.get_nested("prompts", "planner_user_template", default="User Query: {nl_query}\n\n{schema_summary}\n{business_context}\n\n{procedural_knowledge}\n\nGenerate a SQL plan (JSON format) to answer this query.")
        user_prompt = user_template.format(
            nl_query=nl_query,
            schema_summary=schema_summary,
            business_context=business_context,
            procedural_knowledge=procedural_knowledge
        )

        try:
            response = self.llm_client.invoke_with_prompt(system_prompt, user_prompt, response_format="json")
            # Parse JSON from response
            # LLM might wrap in ```json or ```
            response = response.strip()
            if response.startswith('```'):
                # Extract JSON from code block
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
            
            result = json.loads(response)
            
            # Post-process SQL to fix common LLM mistakes
            sql = result.get("sql", "")
            # Fix "SELECT TOP N" -> "SELECT" (remove TOP clause, rely on LIMIT at end)
            import re
            sql = re.sub(r'\bSELECT\s+TOP\s+\d+\s+', 'SELECT ', sql, flags=re.IGNORECASE)
            result["sql"] = sql
            
            return result
        except json.JSONDecodeError as e:
            print(f"⚠ LLM returned invalid JSON: {e}")
            print(f"  Response: {response[:200]}")
            # Fall back to heuristic
            return self._simple_guess(nl_query, available_tables)
        except Exception as e:
            print(f"⚠ LLM planning failed: {e}")
            return self._simple_guess(nl_query, available_tables)

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        nl_query: str = arguments["query"]
        table_schema: Dict[str, Any] = arguments.get("table_schema", {})
        
        # Prefer table_schema from arguments (avoids DuckDB lock issues)
        # Fall back to querying DuckDB only if schema not provided
        if table_schema:
            available_tables = list(table_schema.keys())
        else:
            available_tables = self._list_tables()
        
        # Use LLM if available, otherwise heuristics
        if self.use_llm and self.llm_client:
            print(f"[NLToSQLPlanner] Using LLM to plan query")
            llm_result = self._llm_plan(nl_query, table_schema, available_tables, arguments.get("docs_meta"))
            plan = {
                "type": "sql_plan",
                "sql": llm_result.get("sql", "SELECT 1"),
                "limits": {"preview_rows": self.preview_rows},
                "target_table": llm_result.get("target_table"),
            }
            plan_quality = llm_result.get("plan_quality", "medium")
            plan_explain = llm_result.get("plan_explanation", "LLM-generated SQL plan")
            clarify = llm_result.get("clarification_questions", [])
        else:
            print(f"[NLToSQLPlanner] Using heuristics to plan query")
            plan = self._simple_guess(nl_query, available_tables)
            plan_quality = "low" if plan.get("target_table") is None else "medium"
            plan_explain = "Heuristic plan based on detected table names and default preview limit."
            clarify = []
            if plan_quality == "low":
                clarify.append("Which table should I use?")
        
        return {
            "plan": plan,
            "plan_quality": plan_quality,
            "plan_explain": plan_explain,
            "clarification_questions": clarify,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


