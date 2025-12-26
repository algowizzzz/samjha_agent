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
    query_spec = state.get("query_spec", {}).copy()  # Make a copy to modify
    query_spec_status = state.get("query_spec_status", {}).copy()  # Make a copy to modify
    
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
                        # Update query_spec_status to mark path as verified
                        if "start_table_grain" not in query_spec_status:
                            query_spec_status["start_table_grain"] = {}
                        query_spec_status["start_table_grain"]["status"] = "verified"
                        query_spec_status["start_table_grain"]["source"] = "tool_result"
                        query_spec_status["start_table_grain"]["blocks_execution"] = False
                        logger.info(f"Updated start_table.path to: {entry.get('path')}")
                        break
            
            elif tool_name == "inspect_table":
                # Update grain, dimensions, filters, or time from schema inspection
                # Only fix fields explicitly mentioned in fills_gap (Decider controls what to verify)
                columns = result.get("columns", [])
                if columns:
                    actual_column_names = [c.get("name", "") for c in columns]
                    actual_column_names_lower = [name.lower() for name in actual_column_names]
                    
                    # Helper function to find matching column name
                    def find_matching_column(inferred_name: str) -> str:
                        """Find actual column name that semantically matches inferred name."""
                        inferred_lower = inferred_name.lower()
                        # Exact match (case-insensitive)
                        for col_name in actual_column_names:
                            if col_name.lower() == inferred_lower:
                                return col_name
                        
                        # Semantic matching: find the column that best represents the inferred concept
                        # Strategy: check if actual column name appears as a complete word in inferred name
                        # Prefer longer, more specific matches (e.g., "product_category" → "category" not "product")
                        best_match = None
                        best_score = 0
                        
                        # Normalize inferred name for word boundary checking
                        inferred_normalized = f"_{inferred_lower}_"
                        
                        for col_name in actual_column_names:
                            col_lower = col_name.lower()
                            col_normalized = f"_{col_lower}_"
                            
                            # Check if actual column name appears as a complete word in inferred name
                            # e.g., "product_category" contains "category" as a word (delimited by _ or start/end)
                            if col_normalized in inferred_normalized:
                                # Score based on length (prefer longer, more specific matches)
                                # and how close it is to the end (core concept often at end)
                                score = len(col_lower)
                                
                                # Bonus if it's at the end (core concept)
                                if inferred_lower.endswith(col_lower):
                                    # Check word boundary
                                    if len(inferred_lower) > len(col_lower):
                                        char_before = inferred_lower[-(len(col_lower) + 1)]
                                        if char_before == "_":
                                            score += 10  # Word boundary at end
                                    else:
                                        score += 10  # Exact match already handled above
                                
                                if score > best_score:
                                    best_score = score
                                    best_match = col_name
                            
                            # Fallback: substring match (lowest priority)
                            elif col_lower in inferred_lower:
                                score = len(col_lower) / 2  # Lower score for substring
                                if score > best_score:
                                    best_score = score
                                    best_match = col_name
                        
                        if best_match:
                            return best_match
                        
                        # Return original if no match found (will fail later, but preserve intent)
                        return inferred_name
                    
                    fills_gap_lower = str(fills_gap).lower()
                    
                    # Only process fields mentioned in fills_gap (Decider decides what to verify)
                    
                    # Handle grain (if fills_gap mentions grain)
                    if "grain" in fills_gap_lower or fills_gap == "start_table_grain":
                        if not query_spec.get("grain"):
                            # Simple inference: if we see region, product, etc., infer grain
                            if "region" in actual_column_names_lower:
                                query_spec["grain"] = "one row per region"
                            elif "product" in actual_column_names_lower:
                                query_spec["grain"] = "one row per product"
                    
                    # Handle dimensions (only if fills_gap mentions dimensions)
                    if "dimension" in fills_gap_lower:
                        # Check if fills_gap specifies a particular dimension (e.g., "dimensions.product_category")
                        if "." in fills_gap:
                            # Specific dimension mentioned (e.g., "dimensions.product_category")
                            dim_name = fills_gap.split(".")[-1].strip()
                            if query_spec.get("dimensions"):
                                # Find and fix the specific dimension
                                fixed_dimensions = []
                                for dim in query_spec["dimensions"]:
                                    if dim.lower() == dim_name.lower() or dim_name.lower() in dim.lower():
                                        # This is the dimension to fix
                                        if dim not in actual_column_names:
                                            corrected = find_matching_column(dim)
                                            if corrected != dim:
                                                logger.info(f"Corrected dimension '{dim}' → '{corrected}' (from fills_gap: {fills_gap})")
                                            fixed_dimensions.append(corrected)
                                        else:
                                            fixed_dimensions.append(dim)
                                    else:
                                        fixed_dimensions.append(dim)  # Keep other dimensions unchanged
                                if fixed_dimensions != query_spec["dimensions"]:
                                    query_spec["dimensions"] = fixed_dimensions
                                    # Update status to mark as verified
                                    query_spec_status["dimensions"] = {
                                        "status": "verified",
                                        "source": "tool_result",
                                        "notes": f"Dimensions verified/corrected by inspect_table (from fills_gap: {fills_gap})",
                                        "blocks_execution": False
                                    }
                        else:
                            # General dimensions verification (fix all dimensions)
                            if query_spec.get("dimensions"):
                                fixed_dimensions = []
                                for dim in query_spec["dimensions"]:
                                    if dim in actual_column_names:
                                        fixed_dimensions.append(dim)  # Keep if correct
                                    else:
                                        corrected = find_matching_column(dim)
                                        if corrected != dim:
                                            logger.info(f"Corrected dimension '{dim}' → '{corrected}' (from fills_gap: {fills_gap})")
                                        fixed_dimensions.append(corrected)
                                if fixed_dimensions != query_spec["dimensions"]:
                                    query_spec["dimensions"] = fixed_dimensions
                                    # Update status to mark as verified
                                    query_spec_status["dimensions"] = {
                                        "status": "verified",
                                        "source": "tool_result",
                                        "notes": f"Dimensions verified/corrected by inspect_table (from fills_gap: {fills_gap})",
                                        "blocks_execution": False
                                    }
                            # Add dimensions if missing
                            elif not query_spec.get("dimensions"):
                                dimension_cols = [c.get("name") for c in columns if c.get("name") in ["region", "product", "customer_id", "category"]]
                                if dimension_cols:
                                    query_spec["dimensions"] = dimension_cols
                    
                    # Handle filters (only if fills_gap mentions filter)
                    if "filter" in fills_gap_lower:
                        if query_spec.get("filters"):
                            # Check if specific filter field mentioned
                            if "." in fills_gap:
                                filter_field = fills_gap.split(".")[-1].strip()
                                for filter_item in query_spec["filters"]:
                                    if isinstance(filter_item, dict) and "field" in filter_item:
                                        if filter_item["field"].lower() == filter_field.lower() or filter_field.lower() in filter_item["field"].lower():
                                            if filter_item["field"] not in actual_column_names:
                                                corrected = find_matching_column(filter_item["field"])
                                                if corrected != filter_item["field"]:
                                                    logger.info(f"Corrected filter field '{filter_item['field']}' → '{corrected}' (from fills_gap: {fills_gap})")
                                                    filter_item["field"] = corrected
                            else:
                                # Fix all filter fields
                                for filter_item in query_spec["filters"]:
                                    if isinstance(filter_item, dict) and "field" in filter_item:
                                        field_name = filter_item["field"]
                                        if field_name not in actual_column_names:
                                            corrected = find_matching_column(field_name)
                                            if corrected != field_name:
                                                logger.info(f"Corrected filter field '{field_name}' → '{corrected}' (from fills_gap: {fills_gap})")
                                                filter_item["field"] = corrected
                    
                    # Handle time.column (only if fills_gap mentions time)
                    if "time" in fills_gap_lower and query_spec.get("time") and isinstance(query_spec["time"], dict):
                        time_col = query_spec["time"].get("column", "")
                        if time_col and time_col not in actual_column_names:
                            corrected = find_matching_column(time_col)
                            if corrected != time_col:
                                logger.info(f"Corrected time.column '{time_col}' → '{corrected}' (from fills_gap: {fills_gap})")
                                query_spec["time"]["column"] = corrected
            
            logger.info(f"Step {step_num} completed: {success_condition}")
            
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            # If required gap can't be closed, halt
            if "required" in fills_gap.lower() or "blocks_execution" in success_condition.lower():
                return {"halt_execution": True}
    
    # Return updated query_spec and query_spec_status
    return {"query_spec": query_spec, "query_spec_status": query_spec_status}


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

