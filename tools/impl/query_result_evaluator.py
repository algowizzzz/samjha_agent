"""
Query Result Evaluator Tool
LLM-assisted evaluation of execution results
to determine satisfaction and notes for re-planning or clarification.
"""
from typing import Dict, Any, Optional, Callable
from tools.base_mcp_tool import BaseMCPTool
import json

try:
    from agent.llm_client import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class QueryResultEvaluatorTool(BaseMCPTool):
    def __init__(self, config: Dict = None):
        default_config = {
            "name": "query_result_evaluator",
            "description": "Evaluate query results against original intent",
            "version": "1.0.0",
            "enabled": True,
            "use_llm": True,  # Set to False to force heuristics
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.use_llm = self.config.get("use_llm", True) and LLM_AVAILABLE
        
        # Initialize LLM if available
        self.llm_client = None
        if self.use_llm:
            try:
                self.llm_client = get_llm_client()
                if not self.llm_client.is_available():
                    print("⚠ LLM not available, using heuristics for evaluation")
                    self.use_llm = False
            except Exception as e:
                print(f"⚠ Failed to initialize LLM for evaluator: {e}")
                self.use_llm = False

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "original_query": {"type": "string"},
                "execution_result": {"type": "object"},
                "execution_stats": {"type": "object"},
                "table_schema": {"type": "object"},
            },
            "required": ["original_query", "execution_result"],
        }

    def get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "satisfaction": {"type": "string"},
                "evaluator_notes": {"type": "string"},
                "issues_detected": {"type": "array"},
                "suggested_improvements": {"type": "array"},
            },
            "required": ["satisfaction"],
        }

    def _heuristic_evaluate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic evaluation"""
        result = arguments.get("execution_result") or {}
        rows = result.get("rows", [])
        columns = result.get("columns", [])
        row_count = result.get("row_count", 0)

        issues = []
        suggestions = []
        notes = []

        if not columns:
            issues.append("no_columns")
            suggestions.append("Verify target table and selected columns.")
        if row_count == 0:
            issues.append("empty_result")
            suggestions.append("Adjust filters or clarify intended metric/timeframe.")
        if row_count and row_count > 0 and len(rows) > 0:
            pass

        if issues:
            notes.append("Result did not meet basic expectations.")
            satisfaction = "needs_work"
        else:
            notes.append("Result appears aligned with the request.")
            satisfaction = "satisfied"

        return {
            "satisfaction": satisfaction,
            "evaluator_notes": " ".join(notes),
            "issues_detected": issues,
            "suggested_improvements": suggestions,
        }

    def _llm_evaluate(self, arguments: Dict[str, Any], stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """LLM-based evaluation"""
        original_query = arguments.get("original_query", "")
        execution_result = arguments.get("execution_result", {})
        execution_stats = arguments.get("execution_stats", {})
        
        # Get prompts from config if available, otherwise use defaults
        from agent.config import QueryAgentConfig
        cfg = QueryAgentConfig()
        
        system_prompt = cfg.get_nested("prompts", "evaluator_system", default="""You are a query result evaluator. Assess if the query results satisfactorily answer the user's original question.

Evaluate based on:
1. Does the result contain relevant data?
2. Is the result empty when it shouldn't be?
3. Do the columns match what the user likely wanted?
4. Are there any obvious errors or issues?

Respond with JSON:
{
  "satisfaction": "satisfied|needs_work|failed",
  "evaluator_notes": "Brief explanation of your assessment",
  "issues_detected": ["issue1", "issue2"],
  "suggested_improvements": ["suggestion1", "suggestion2"]
}""")

        # Prepare result preview (limit to avoid token overflow)
        result_preview = {
            "row_count": execution_result.get("row_count", 0),
            "columns": execution_result.get("columns", []),
            "sample_rows": execution_result.get("rows", [])[:5],  # First 5 rows only
        }
        
        # Get user prompt template from config
        user_template = cfg.get_nested("prompts", "evaluator_user_template", default="Original User Query: {original_query}\n\nExecution Result:\n{result_preview}\n\nExecution Stats:\n{execution_stats}\n\nEvaluate if this result satisfactorily answers the user's query.")
        user_prompt = user_template.format(
            original_query=original_query,
            result_preview=json.dumps(result_preview, indent=2),
            execution_stats=json.dumps(execution_stats, indent=2)
        )

        try:
            # Use streaming if callback provided, otherwise use regular invoke
            if stream_callback:
                # Stream and accumulate full response
                full_response = ""
                for chunk in self.llm_client.stream_with_prompt(
                    system_prompt, user_prompt, response_format="json", callback=stream_callback
                ):
                    full_response += chunk
                response = full_response
            else:
                response = self.llm_client.invoke_with_prompt(system_prompt, user_prompt, response_format="json")
            
            # Parse JSON
            response = response.strip()
            if response.startswith('```'):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
            
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            print(f"⚠ LLM returned invalid JSON: {e}")
            return self._heuristic_evaluate(arguments)
        except Exception as e:
            print(f"⚠ LLM evaluation failed: {e}")
            return self._heuristic_evaluate(arguments)

    def execute(self, arguments: Dict[str, Any], stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        if self.use_llm and self.llm_client:
            print(f"[QueryResultEvaluator] Using LLM to evaluate results")
            return self._llm_evaluate(arguments, stream_callback=stream_callback)
        else:
            print(f"[QueryResultEvaluator] Using heuristics to evaluate results")
            return self._heuristic_evaluate(arguments)


