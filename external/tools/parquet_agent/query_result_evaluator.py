"""
Query Result Evaluator Tool - Adapted for spec schema.
Evaluates if results satisfy QuerySpec.
"""

from typing import Dict, Any, Optional
from tools.base_mcp_tool import BaseMCPTool
import json

try:
    from external.platform.llm import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

from external.platform.prompt_loader import load_prompt


class QueryResultEvaluatorTool(BaseMCPTool):
    def __init__(self, config: Dict = None):
        default_config = {
            "name": "query_result_evaluator",
            "description": "Evaluate query results against QuerySpec",
            "version": "2.0.0",  # Updated version for spec compliance
            "enabled": True,
            "use_llm": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.use_llm = self.config.get("use_llm", True) and LLM_AVAILABLE
        
        self.llm_client = None
        if self.use_llm:
            try:
                self.llm_client = get_llm_client()
                if not self.llm_client.is_available():
                    self.use_llm = False
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM for evaluator: {e}")
                self.use_llm = False

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["query_spec", "results_summary", "validation_checks"],
            "properties": {
                "query_spec": {"type": "object"},
                "results_summary": {"type": "object"},
                "validation_checks": {"type": "array"}
            }
        }

    def get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["satisfied", "issues", "notes"],
            "properties": {
                "satisfied": {"type": "boolean"},
                "issues": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"}
            }
        }

    def _heuristic_evaluate(self, query_spec: dict, results_summary: dict, validation_checks: list) -> Dict[str, Any]:
        """Fallback heuristic evaluation"""
        issues = []
        notes = []
        satisfied = True
        
        # Output shape check
        expected_columns = query_spec.get("output_shape", {}).get("columns", [])
        if expected_columns:
            actual_columns = results_summary.get("column_names", [])
            missing = [col for col in expected_columns if col not in actual_columns]
            if missing:
                issues.append(f"Missing columns: {', '.join(missing)}")
                satisfied = False
        
        # Row count check
        row_count = results_summary.get("row_count", 0)
        if row_count == 0:
            # Check if validation_checks mention zero rows as an issue
            for check in validation_checks:
                if "zero" in str(check).lower() or "empty" in str(check).lower():
                    issues.append("Zero rows returned")
                    satisfied = False
                    break
        
        # Apply validation_checks
        for check in validation_checks:
            check_str = str(check).lower()
            # Simple heuristic: if check mentions row_count > 0 and we have 0, fail
            if "row_count" in check_str and ">" in check_str and row_count == 0:
                issues.append(f"Validation check failed: {check}")
                satisfied = False
        
        if not issues:
            notes.append("Results appear to satisfy the Query Spec")
        else:
            notes.append(f"Found {len(issues)} issue(s)")
        
        return {
            "satisfied": satisfied,
            "issues": issues,
            "notes": " ".join(notes)
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # #region agent log
        import json
        query_spec = arguments["query_spec"]
        results_summary = arguments["results_summary"]
        validation_checks = arguments.get("validation_checks", [])
        
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"query_result_evaluator.py:execute:entry","message":"Query result evaluator entry","data":{"query_spec_grain":query_spec.get("grain",""),"row_count":results_summary.get("row_count",0),"column_count":len(results_summary.get("column_names",[]))},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        if self.use_llm and self.llm_client:
            try:
                # Serialize results_summary with timestamp conversion
                serialized_results = {
                    "row_count": results_summary.get("row_count", 0),
                    "column_names": results_summary.get("column_names", []),
                    "sample_rows": []
                }
                for row in results_summary.get("sample_rows", []):
                    serialized_row = {}
                    for key, value in row.items():
                        if hasattr(value, 'isoformat'):  # Timestamp/datetime objects
                            serialized_row[key] = value.isoformat()
                        else:
                            serialized_row[key] = value
                    serialized_results["sample_rows"].append(serialized_row)
                
                prompt_template = load_prompt("query_result_evaluator")
                prompt = prompt_template + f"""

Query Spec:
{json.dumps(query_spec, indent=2)}

Results Summary:
{json.dumps(serialized_results, indent=2)}

Validation Checks:
{json.dumps(validation_checks, indent=2)}

Output JSON only:
"""
                response = self.llm_client.invoke_with_prompt(
                    system_prompt="",
                    user_prompt=prompt,
                    response_format="json"
                )
                
                # Extract JSON from response
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                elif response.startswith("```json"):
                    response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                
                result = json.loads(response)
                # #region agent log
                with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"query_result_evaluator.py:execute:llm_result","message":"LLM evaluation result","data":{"satisfied":result.get("satisfied"),"issues":result.get("issues",[]),"notes":result.get("notes","")[:200]},"timestamp":int(__import__('time').time()*1000)})+'\n')
                # #endregion
                return result
            except Exception as e:
                self.logger.warning(f"LLM evaluation failed: {e}, falling back to heuristics")
                # #region agent log
                with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"query_result_evaluator.py:execute:llm_failed","message":"LLM evaluation failed, using heuristics","data":{"error":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n')
                # #endregion
        
        # Fallback to heuristics
        heuristic_result = self._heuristic_evaluate(query_spec, results_summary, validation_checks)
        # #region agent log
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"query_result_evaluator.py:execute:heuristic_result","message":"Heuristic evaluation result","data":{"satisfied":heuristic_result.get("satisfied"),"issues":heuristic_result.get("issues",[]),"notes":heuristic_result.get("notes","")[:200]},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        return heuristic_result

