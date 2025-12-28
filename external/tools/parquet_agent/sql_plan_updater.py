"""
SQL Plan Updater Tool - Apply minimal patches to SQL.
"""

from tools.base_mcp_tool import BaseMCPTool

try:
    from external.platform.llm import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

from external.platform.prompt_loader import load_prompt


class SQLPlanUpdaterTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "sql_plan_updater",
            "description": "Apply minimal patches to SQL",
            "version": "1.0.0",
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
                self.logger.warning(f"Failed to initialize LLM for SQL updater: {e}")
                self.llm_client = None

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["sql", "patch_instructions"],
            "properties": {
                "sql": {"type": "string"},
                "patch_instructions": {"type": "string"}
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
        sql = arguments["sql"]
        patch_instructions = arguments["patch_instructions"]
        
        if not self.llm_client:
            raise ValueError("LLM client not available for SQL patching")
        
        prompt_template = load_prompt("sql_plan_updater")
        prompt = prompt_template.format(
            sql=sql,
            patch_instructions=patch_instructions
        )
        
        response = self.llm_client.invoke_with_prompt(
            system_prompt="",
            user_prompt=prompt,
            response_format=None
        )
        patched_sql = response.strip()
        
        # Clean markdown if present
        if patched_sql.startswith("```"):
            patched_sql = patched_sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        elif patched_sql.startswith("```sql"):
            patched_sql = patched_sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        return {"sql": patched_sql}

