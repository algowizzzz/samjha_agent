"""
Executor Nodes - 6 nodes for linear Executor graph.
"""

import logging
from typing import Dict, Any
from external.agent.state_types import ExecutorState
from external.agent.sql_gate import spec_ready_for_sql
from external.agent.schema_validators import validate_executor_report

logger = logging.getLogger(__name__)


def investigation_node(state: ExecutorState, tools_registry) -> Dict[str, Any]:
    """
    InvestigationNode - Runs investigation plan steps.
    
    Input: investigation_plan, query_spec, query_spec_status
    Output: Updated query_spec and query_spec_status
    """
    if state.get("halt_execution", False):
        return {}
    
    investigation_plan = state.get("investigation_plan", [])
    query_spec = state.get("query_spec", {})
    query_spec_status = state.get("query_spec_status", {})
    
    logger.info(f"InvestigationNode: Running {len(investigation_plan)} plan steps")
    
    # Execute each step in the plan
    for step in investigation_plan:
        step_num = step.get("step", 0)
        tool_name = step.get("tool")
        args = step.get("args", {})
        fills_gap = step.get("fills_gap", "")
        success_condition = step.get("success_condition", "")
        
        logger.info(f"Step {step_num}: {tool_name} - {fills_gap}")
        
        try:
            # Get tool from registry
            tool = tools_registry.get_tool(tool_name)
            if not tool:
                logger.error(f"Tool not found: {tool_name}")
                return {"halt_execution": True}
            
            # Execute tool
            result = tool.execute(args)
            
            # Patch query_spec and query_spec_status based on result
            if tool_name == "list_dir" and fills_gap == "start_table.path":
                # Find matching file based on start_table.name
                entries = result.get("entries", [])
                table_name = query_spec.get("start_table", {}).get("name", "")
                for entry in entries:
                    if entry.get("type") == "file" and table_name in entry.get("name", ""):
                        # Update query_spec with found path
                        if "start_table" not in query_spec:
                            query_spec["start_table"] = {}
                        query_spec["start_table"]["path"] = entry.get("path", "")
                        logger.info(f"Updated start_table.path to: {entry.get('path')}")
                        break
            
            elif tool_name == "inspect_table" and fills_gap in ["start_table_grain", "grain", "dimensions"]:
                # Update grain or dimensions from schema inspection
                columns = result.get("columns", [])
                if columns:
                    # If grain was missing, infer from columns
                    if not query_spec.get("grain"):
                        # Simple inference: if we see region, product, etc., infer grain
                        if "region" in [c.get("name", "").lower() for c in columns]:
                            query_spec["grain"] = "one row per region"
                        elif "product" in [c.get("name", "").lower() for c in columns]:
                            query_spec["grain"] = "one row per product"
                    # Update dimensions if missing
                    if not query_spec.get("dimensions"):
                        dimension_cols = [c.get("name") for c in columns if c.get("name") in ["region", "product", "customer_id"]]
                        if dimension_cols:
                            query_spec["dimensions"] = dimension_cols
            
            logger.info(f"Step {step_num} completed: {success_condition}")
            
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            # If required gap can't be closed, halt
            if "required" in fills_gap.lower() or "blocks_execution" in success_condition.lower():
                return {"halt_execution": True}
    
    # Return updated query_spec
    return {"query_spec": query_spec}


def sql_generation_node(state: ExecutorState, tools_registry, domain_md: str) -> Dict[str, Any]:
    """
    SQLGenerationNode - Generates SQL from completed QuerySpec.
    
    Gate: Must check spec_ready_for_sql() first.
    """
    if state.get("halt_execution", False):
        return {}
    
    query_spec = state.get("query_spec", {})
    query_spec_status = state.get("query_spec_status", {})
    
    # SQL Generation Gate
    if not spec_ready_for_sql(query_spec, query_spec_status, domain_md):
        logger.error("QuerySpec not ready for SQL generation")
        return {"halt_execution": True}
    
    logger.info("SQLGenerationNode: Generating SQL from QuerySpec")
    
    try:
        tool = tools_registry.get_tool("nl_to_sql_planner")
        if not tool:
            logger.error("nl_to_sql_planner tool not found")
            return {"halt_execution": True}
        
        result = tool.execute({
            "query_spec": query_spec,
            "query_spec_status": query_spec_status
        })
        
        sql = result.get("sql", "")
        if sql.startswith("ERROR:"):
            logger.error(f"SQL generation error: {sql}")
            return {"halt_execution": True}
        
        return {"final_sql": sql}
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return {"halt_execution": True}


def safety_validation_node(state: ExecutorState, tools_registry) -> Dict[str, Any]:
    """
    SafetyValidationNode - Validates SQL for safety.
    """
    if state.get("halt_execution", False):
        return {}
    
    final_sql = state.get("final_sql")
    policy_limits = state.get("policy_limits", {})
    
    if not final_sql:
        logger.error("No SQL to validate")
        return {"halt_execution": True}
    
    logger.info("SafetyValidationNode: Validating SQL")
    
    try:
        tool = tools_registry.get_tool("query_safety_validator")
        if not tool:
            logger.error("query_safety_validator tool not found")
            return {"halt_execution": True}
        
        result = tool.execute({
            "sql": final_sql,
            "policy_limits": policy_limits
        })
        
        if not result.get("allowed", False):
            logger.error(f"SQL validation failed: {result.get('flags', [])}")
            return {"halt_execution": True}
        
        return {}
        
    except Exception as e:
        logger.error(f"Safety validation failed: {e}")
        return {"halt_execution": True}


def execution_node(state: ExecutorState, tools_registry) -> Dict[str, Any]:
    """
    ExecutionNode - Executes SQL and captures results.
    """
    if state.get("halt_execution", False):
        return {}
    
    final_sql = state.get("final_sql")
    policy_limits = state.get("policy_limits", {})
    
    if not final_sql:
        logger.error("No SQL to execute")
        return {"halt_execution": True}
    
    logger.info("ExecutionNode: Executing SQL")
    
    try:
        tool = tools_registry.get_tool("execute_sql")
        if not tool:
            logger.error("execute_sql tool not found")
            return {"halt_execution": True}
        
        result = tool.execute({
            "sql": final_sql,
            "timeout_seconds": policy_limits.get("timeout_seconds"),
            "max_rows": policy_limits.get("max_rows")
        })
        
        return {"results": result}
        
    except Exception as e:
        logger.error(f"SQL execution failed: {e}")
        return {"halt_execution": True, "last_error": str(e)}


def evaluation_node(state: ExecutorState, tools_registry) -> Dict[str, Any]:
    """
    EvaluationNode - Evaluates results against QuerySpec.
    """
    # #region agent log
    import json
    with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"executor_nodes.py:evaluation_node:entry","message":"Evaluation node entry","data":{"halt_execution":state.get("halt_execution",False),"has_results":state.get("results") is not None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    if state.get("halt_execution", False):
        # #region agent log
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"executor_nodes.py:evaluation_node:halt","message":"Execution halted, returning empty","data":{},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        return {}
    
    query_spec = state.get("query_spec", {})
    results = state.get("results")
    
    if not results:
        logger.error("No results to evaluate")
        # #region agent log
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"executor_nodes.py:evaluation_node:no_results","message":"No results to evaluate, halting","data":{},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        return {"halt_execution": True}
    
    logger.info("EvaluationNode: Evaluating results")
    
    try:
        tool = tools_registry.get_tool("query_result_evaluator")
        if not tool:
            logger.error("query_result_evaluator tool not found")
            # #region agent log
            with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"executor_nodes.py:evaluation_node:tool_not_found","message":"query_result_evaluator tool not found","data":{},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            return {"halt_execution": True}
        
        # Build results_summary from results
        results_summary = {
            "row_count": results.get("row_count", 0),
            "column_names": results.get("columns", []),
            "sample_rows": results.get("rows_preview", [])
        }
        
        # #region agent log
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"executor_nodes.py:evaluation_node:before_tool","message":"Before query_result_evaluator call","data":{"query_spec_grain":query_spec.get("grain",""),"row_count":results_summary["row_count"],"column_count":len(results_summary["column_names"])},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        evaluation = tool.execute({
            "query_spec": query_spec,
            "results_summary": results_summary,
            "validation_checks": query_spec.get("validation_checks", [])
        })
        
        # #region agent log
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"executor_nodes.py:evaluation_node:after_tool","message":"After query_result_evaluator call","data":{"evaluation":evaluation,"satisfied":evaluation.get("satisfied") if isinstance(evaluation,dict) else None,"has_issues":bool(evaluation.get("issues",[])) if isinstance(evaluation,dict) else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        return {"evaluation": evaluation}
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # #region agent log
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"executor_nodes.py:evaluation_node:exception","message":"Evaluation exception","data":{"error":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        return {"halt_execution": True}


def outcome_node(state: ExecutorState) -> Dict[str, Any]:
    """
    OutcomeNode - Builds ExecutorReport (SUCCESS or ERROR).
    """
    # #region agent log
    import json
    halt_execution = state.get("halt_execution", False)
    evaluation = state.get("evaluation") or {}
    final_sql = state.get("final_sql", "")
    results = state.get("results")
    last_error = state.get("last_error", "")
    
    with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"executor_nodes.py:outcome_node:entry","message":"Outcome node entry","data":{"halt_execution":halt_execution,"evaluation_keys":list(evaluation.keys()) if isinstance(evaluation,dict) else "not_dict","evaluation_satisfied":evaluation.get("satisfied") if isinstance(evaluation,dict) else None,"has_last_error":bool(last_error)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    if halt_execution or not evaluation.get("satisfied", False):
        # Build ERROR report
        error_type = "SQL" if last_error else "GRAIN" if not evaluation.get("satisfied") else "SCHEMA"
        
        # #region agent log
        with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"executor_nodes.py:outcome_node:error","message":"Building ERROR report","data":{"error_type":error_type,"halt_execution":halt_execution,"evaluation_satisfied":evaluation.get("satisfied") if isinstance(evaluation,dict) else None,"evaluation_issues":evaluation.get("issues",[]) if isinstance(evaluation,dict) else [],"evaluation_notes":evaluation.get("notes","")[:200] if isinstance(evaluation,dict) else ""},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        executor_report = {
            "status": "ERROR",
            "error_type": error_type,
            "failed_checklist_items": evaluation.get("issues", []),
            "what_changed": "Execution halted",
            "minimal_fix_suggestion": evaluation.get("notes", ""),
            "last_sql": final_sql or "",
            "last_error": last_error or "Execution halted"
        }
    else:
        # Build SUCCESS report
        result_summary = f"Returned {results.get('row_count', 0)} rows with {len(results.get('columns', []))} columns"
        finished_output = f"Query executed successfully. {result_summary}."
        
        executor_report = {
            "status": "SUCCESS",
            "final_sql": final_sql,
            "result_summary": result_summary,
            "evaluation": evaluation,
            "finished_output": finished_output
        }
    
    # Validate report
    valid, error = validate_executor_report(executor_report)
    if not valid:
        logger.error(f"Invalid executor report: {error}")
        # Create a minimal error report
        executor_report = {
            "status": "ERROR",
            "error_type": "SCHEMA",
            "failed_checklist_items": ["executor_report_validation"],
            "what_changed": "Report validation failed",
            "minimal_fix_suggestion": error or "Unknown error",
            "last_sql": final_sql or "",
            "last_error": error or "Report validation failed"
        }
    
    return {"executor_report": executor_report}

