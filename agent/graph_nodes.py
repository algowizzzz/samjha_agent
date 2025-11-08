import json
import os
import glob
import time
from datetime import datetime
from typing import Dict, Any

from agent.schemas import AgentState, PartialState
from agent.config import QueryAgentConfig
from agent.state_manager import AgentStateManager
from tools.impl.nl_to_sql_planner import NLToSQLPlannerTool
from tools.impl.query_safety_validator import QuerySafetyValidatorTool
from tools.impl.query_result_evaluator import QueryResultEvaluatorTool
import hashlib

try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def invoke_node(state: AgentState, cfg: QueryAgentConfig, sm: AgentStateManager) -> PartialState:
    user_id = state.get("user_id")
    parquet_location = cfg.get_nested("parquet_paths", default=["data/duckdb"])[0]
    logs = state.get("logs", [])
    logs.append({"node": "invoke", "timestamp": _now_iso(), "msg": "invoke started"})
    
    # Initialize or load conversation history (short-term memory within session)
    conversation_history = state.get("conversation_history", [])
    
    # Populate table_schema from available CSV/Parquet files
    # (DuckDB views are temporary per-connection, so we infer schema from files)
    table_schema = {}
    docs_meta = []
    try:
        import os
        import glob
        import csv
        import json
        
        # Load data dictionary for business context
        data_dict_path = "config/data_dictionary.json"
        data_dictionary = {}
        if os.path.exists(data_dict_path):
            try:
                with open(data_dict_path, 'r') as f:
                    data_dictionary = json.load(f)
            except:
                pass
        
        data_files = glob.glob(f"{parquet_location}/*.csv") + glob.glob(f"{parquet_location}/*.parquet")
        for file_path in data_files:
            filename = os.path.basename(file_path)
            table_name = os.path.splitext(filename)[0]  # Remove extension
            
            # Read column names from CSV header
            columns = []
            if file_path.endswith('.csv'):
                try:
                    with open(file_path, 'r') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        columns = [{"name": col, "type": "VARCHAR"} for col in header]
                except:
                    pass
            
            table_schema[table_name] = {"columns": columns}
            
            # Add business context from data dictionary
            if table_name in data_dictionary.get("tables", {}):
                table_info = data_dictionary["tables"][table_name]
                docs_meta.append({
                    "table": table_name,
                    "description": table_info.get("description", ""),
                    "business_context": table_info.get("business_context", ""),
                    "key_columns": table_info.get("key_columns", {})
                })
        
        # Add business glossary to docs_meta
        if "business_glossary" in data_dictionary:
            docs_meta.append({
                "type": "business_glossary",
                "glossary": data_dictionary["business_glossary"]
            })
        
        logs.append({"node": "invoke", "timestamp": _now_iso(), "msg": f"loaded schema for {len(table_schema)} tables, {len(docs_meta)} metadata entries"})
            
    except Exception as e:
        logs.append({"node": "invoke", "timestamp": _now_iso(), "msg": f"error loading table schema: {str(e)}", "level": "error"})
        print(f"[invoke_node] Could not load table schema: {e}")
    
    return {
        "last_node": "invoke",
        "control": "plan",
        "parquet_location": parquet_location,
        "conversation_history": conversation_history,
        "table_schema": table_schema,
        "docs_meta": docs_meta,
        "logs": logs,
        "timestamp": _now_iso(),
    }


def planner_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    logs = state.get("logs", [])
    logs.append({"node": "planner", "timestamp": _now_iso(), "msg": "planning query"})
    
    try:
        planner = NLToSQLPlannerTool({
            "preview_rows": cfg.get_nested("duckdb", "preview_rows", default=100),
            "data_directory": state.get("parquet_location", "data/duckdb"),
            "use_llm": True,  # Enable LLM for planning
        })
        res = planner.execute({
            "query": state["user_input"],
            "table_schema": state.get("table_schema", {}),
            "docs_meta": state.get("docs_meta", []),
            "previous_clarifications": [],
        })
        plan = res["plan"]
        plan_quality = res.get("plan_quality", "low")
        plan_explain = res.get("plan_explain", "")
        clarification_questions = res.get("clarification_questions", [])
        control = "execute" if plan_quality == "high" else "clarify"
        
        # compute plan_id
        plan_sql = (plan or {}).get("sql", "")
        plan_hash = hashlib.sha256((state.get("user_input", "") + "|" + plan_sql).encode("utf-8")).hexdigest()[:16]
        metrics = state.get("metrics") or {}
        metrics["plan_id"] = plan_hash
        
        logs.append({"node": "planner", "timestamp": _now_iso(), 
                    "msg": f"plan generated: quality={plan_quality}, control={control}, clarifications={len(clarification_questions)}"})
        
        return {
            "last_node": "planner",
            "plan": plan,
            "plan_quality": plan_quality,
            "plan_explain": plan_explain,
            "clarification_questions": clarification_questions,
            "control": control,
            "metrics": metrics,
            "logs": logs,
        }
    except Exception as e:
        logs.append({"node": "planner", "timestamp": _now_iso(), "msg": f"planning failed: {str(e)}", "level": "error"})
        return {
            "last_node": "planner",
            "plan": None,
            "plan_quality": "error",
            "plan_explain": f"Planning error: {str(e)}",
            "control": "end",
            "logs": logs,
        }


def clarify_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    logs = state.get("logs", [])
    logs.append({"node": "clarify", "timestamp": _now_iso(), "msg": "requesting user clarification"})
    
    try:
        # Get clarification questions from planner result
        plan_explain = state.get("plan_explain", "")
        plan_quality = state.get("plan_quality", "low")
        user_input = state.get("user_input", "")
        clarification_questions = state.get("clarification_questions", [])
        
        # Build specific clarification questions with reasoning
        clarify_reasoning = []
        
        # Generate reasoning based on plan quality and explanation
        if plan_quality == "low":
            clarify_reasoning.append("The query is ambiguous and needs more context")
        elif plan_quality == "medium":
            clarify_reasoning.append("The query could be interpreted in multiple ways")
        
        # Add specific reasons from plan explanation
        plan_explain_lower = plan_explain.lower()
        if "multiple tables" in plan_explain_lower or "table" in plan_explain_lower:
            clarify_reasoning.append("Multiple tables might match your request")
        if "column" in plan_explain_lower:
            clarify_reasoning.append("Column names or data types are unclear")
        if "ambiguous" in plan_explain_lower:
            clarify_reasoning.append("The intent of your query is ambiguous")
        if "uncertain" in plan_explain_lower:
            clarify_reasoning.append("I'm uncertain about the best way to answer this")
        
        # If no specific questions from planner, generate generic ones
        if not clarification_questions:
            clarification_questions = [
                "Which specific table or dataset should I query?",
                "What columns or metrics are you interested in?",
                "Are there any filters or conditions I should apply?"
            ]
        
        # Build clarification prompt with questions and reasoning
        reasoning_text = "\n".join(f"• {reason}" for reason in clarify_reasoning) if clarify_reasoning else "• I need more information to generate an accurate query"
        questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(clarification_questions))
        
        clarify_prompt = f"""I need clarification on your query: "{user_input}"

**Why I'm asking:**
{reasoning_text}

**Questions:**
{questions_text}

Please provide more details to help me generate an accurate query."""
        
        # Update metrics
        metrics = state.get("metrics") or {}
        metrics["clarify_turns"] = metrics.get("clarify_turns", 0) + 1
        
        logs.append({"node": "clarify", "timestamp": _now_iso(), 
                    "msg": f"clarification prompt generated with {len(clarification_questions)} questions (turn {metrics['clarify_turns']})"})
        
        return {
            "last_node": "clarify",
            "clarify_prompt": clarify_prompt,
            "clarify_questions": clarification_questions,
            "clarify_reasoning": clarify_reasoning,
            "control": "wait_for_user",
            "metrics": metrics,
            "logs": logs,
        }
    except Exception as e:
        logs.append({"node": "clarify", "timestamp": _now_iso(), "msg": f"clarify failed: {str(e)}", "level": "error"})
        return {
            "last_node": "clarify",
            "control": "end",
            "logs": logs,
        }


def replan_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    logs = state.get("logs", [])
    logs.append({"node": "replan", "timestamp": _now_iso(), "msg": "replanning with user clarification"})
    
    try:
        # Combine original query with user clarification
        original_query = state["user_input"]
        user_clarification = state.get("user_clarification") or ""
        combined = f"{original_query}\n\nAdditional context: {user_clarification}"
        
        logs.append({"node": "replan", "timestamp": _now_iso(), 
                    "msg": f"replanning with clarification: {user_clarification[:50]}..."})
        
        planner = NLToSQLPlannerTool({
            "preview_rows": cfg.get_nested("duckdb", "preview_rows", default=100),
            "data_directory": state.get("parquet_location", "data/duckdb"),
            "use_llm": True,
        })
        res = planner.execute({
            "query": combined,
            "table_schema": state.get("table_schema", {}),
            "docs_meta": state.get("docs_meta", []),
            "previous_clarifications": [user_clarification],
        })
        plan = res["plan"]
        plan_quality = res.get("plan_quality", "low")
        clarification_questions = res.get("clarification_questions", [])
        
        logs.append({"node": "replan", "timestamp": _now_iso(), 
                    "msg": f"replan complete: quality={plan_quality}"})
        
        return {
            "last_node": "replan",
            "plan": plan,
            "plan_quality": plan_quality,
            "plan_explain": res.get("plan_explain", ""),
            "clarification_questions": clarification_questions,
            "control": "execute" if plan_quality == "high" else "clarify",
            "logs": logs,
        }
    except Exception as e:
        logs.append({"node": "replan", "timestamp": _now_iso(), 
                    "msg": f"replan failed: {str(e)}", "level": "error"})
        return {
            "last_node": "replan",
            "control": "end",
            "logs": logs,
        }


def execute_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    logs = state.get("logs", [])
    logs.append({"node": "execute", "timestamp": _now_iso(), "msg": "executing query"})
    
    if duckdb is None:
        logs.append({"node": "execute", "timestamp": _now_iso(), "msg": "DuckDB not installed", "level": "error"})
        return {
            "last_node": "execute",
            "execution_result": None,
            "execution_stats": {"error": "DuckDB not installed", "execution_time_ms": 0},
            "control": "end",
            "logs": logs,
        }
    
    try:
        safety = QuerySafetyValidatorTool({
            "limits": cfg.get("limits") or {"max_rows": 1000},
            "safety": cfg.get("safety") or {},
        })
        preview_rows = int(cfg.get_nested("duckdb", "preview_rows", default=100))
        plan_sql = (state.get("plan") or {}).get("sql", "SELECT 1")
        
        logs.append({"node": "execute", "timestamp": _now_iso(), "msg": f"validating query: {plan_sql[:50]}..."})
        
        safe = safety.execute({
            "query": plan_sql,
            "enforce_limit": True,
            "default_limit": preview_rows,
        })
        if not safe.get("is_safe", False):
            error_msg = safe.get("reason", "unsafe query")
            logs.append({"node": "execute", "timestamp": _now_iso(), "msg": f"query validation failed: {error_msg}", "level": "error"})
            return {
                "last_node": "execute",
                "execution_result": None,
                "execution_stats": {"error": error_msg, "execution_time_ms": 0, "limited": False},
                "control": "end",
                "logs": logs,
            }
        sql = safe.get("sanitized_query", plan_sql)
        
        logs.append({"node": "execute", "timestamp": _now_iso(), "msg": "query validated, executing..."})
        
        # Execute query directly with DuckDB (standalone mode)
        data_dir = state.get('parquet_location', 'data/duckdb')
        
        start_time = time.time()
        conn = duckdb.connect(':memory:', read_only=False)
        
        # Load CSV/Parquet files as views
        for csv_file in glob.glob(f"{data_dir}/*.csv"):
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto('{csv_file}')")
        
        for parquet_file in glob.glob(f"{data_dir}/*.parquet"):
            table_name = os.path.splitext(os.path.basename(parquet_file))[0]
            conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{parquet_file}')")
        
        # Execute query
        result = conn.execute(sql).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []
        conn.close()
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Format result
        result_payload = {
            "query": sql,
            "columns": columns,
            "rows": [dict(zip(columns, row)) for row in result],
            "row_count": len(result),
            "execution_time_ms": execution_time_ms,
            "limited": len(result) >= preview_rows
        }
        stats = {
            "limited": safe.get("limit_enforced", False),
            "execution_time_ms": result_payload.get("execution_time_ms", 0),
            "error": None,
        }
        # update metrics
        metrics = state.get("metrics") or {}
        metrics["row_count"] = result_payload["row_count"]
        metrics["column_count"] = len(result_payload["columns"])
        
        logs.append({"node": "execute", "timestamp": _now_iso(), "msg": f"query executed successfully: {result_payload['row_count']} rows returned"})
        
        return {
            "last_node": "execute",
            "execution_result": result_payload,
            "execution_stats": stats,
            "control": "evaluate",
            "metrics": metrics,
            "logs": logs,
        }
    except Exception as e:
        error_msg = str(e)
        logs.append({"node": "execute", "timestamp": _now_iso(), "msg": f"query execution failed: {error_msg}", "level": "error"})
        return {
            "last_node": "execute",
            "execution_result": None,
            "execution_stats": {"error": error_msg, "execution_time_ms": 0, "limited": False},
            "control": "end",
            "logs": logs,
        }


def evaluate_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    logs = state.get("logs", [])
    logs.append({"node": "evaluate", "timestamp": _now_iso(), "msg": "evaluating query results"})
    
    try:
        # Check if execution failed
        execution_stats = state.get("execution_stats", {})
        if execution_stats.get("error"):
            logs.append({"node": "evaluate", "timestamp": _now_iso(), "msg": f"skipping evaluation due to execution error: {execution_stats['error']}", "level": "warning"})
            return {
                "last_node": "evaluate",
                "satisfaction": "failed",
                "evaluator_notes": f"Query execution failed: {execution_stats['error']}",
                "control": "end",
                "logs": logs,
            }
        
        evaluator = QueryResultEvaluatorTool({"use_llm": True})  # Enable LLM for evaluation
        res = evaluator.execute({
            "original_query": state["user_input"],
            "execution_result": state.get("execution_result", {}),
            "execution_stats": execution_stats,
            "table_schema": state.get("table_schema", {}),
        })
        satisfaction = res.get("satisfaction", "needs_work")
        control = "end" if satisfaction == "satisfied" else "replan"
        
        logs.append({"node": "evaluate", "timestamp": _now_iso(), "msg": f"evaluation complete: satisfaction={satisfaction}, control={control}"})
        
        return {
            "last_node": "evaluate",
            "satisfaction": satisfaction,
            "evaluator_notes": res.get("evaluator_notes", ""),
            "control": control,
            "logs": logs,
        }
    except Exception as e:
        logs.append({"node": "evaluate", "timestamp": _now_iso(), "msg": f"evaluation failed: {str(e)}", "level": "error"})
        return {
            "last_node": "evaluate",
            "satisfaction": "failed",
            "evaluator_notes": f"Evaluation error: {str(e)}",
            "control": "end",
            "logs": logs,
        }


def end_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    logs = state.get("logs", [])
    logs.append({"node": "end", "timestamp": _now_iso(), "msg": "finalizing agent response"})
    
    # Calculate total duration
    metrics = state.get("metrics", {})
    start_time_str = metrics.get("start_time")
    total_ms = 0
    if start_time_str:
        try:
            from datetime import datetime
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            end_time = datetime.utcnow()
            total_ms = int((end_time - start_time).total_seconds() * 1000)
            metrics["total_ms"] = total_ms
        except:
            pass
    
    # Build dual outputs
    # response: concise preview
    preview_rows = int(cfg.get_nested("duckdb", "preview_rows", default=100))
    exec_result = state.get("execution_result") or {}
    rows = exec_result.get("rows", [])[:preview_rows]
    columns = exec_result.get("columns", [])
    execution_stats = state.get("execution_stats", {})
    
    # Build response - user-friendly summary
    response_lines = []
    if execution_stats.get("error"):
        response_lines.append(f"❌ Query failed: {execution_stats['error']}")
        response_lines.append(f"\nQuery attempted: {state.get('plan', {}).get('sql', '')}")
    else:
        # Generate natural language summary based on the query and results
        user_query = state.get("user_input", "")
        row_count = exec_result.get("row_count", 0)
        
        if row_count == 0:
            response_lines.append(f"No results found for: {user_query}")
        else:
            # Create a summary based on the data
            response_lines.append(f"Here are the results for '{user_query}':\n")
            
            # Show the data in a readable format
            if rows and len(rows) > 0:
                # Format as a table-like structure
                for i, row in enumerate(rows[:10], 1):  # Show up to 10 rows
                    row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                    response_lines.append(f"{i}. {row_str}")
                
                if row_count > len(rows[:10]):
                    response_lines.append(f"\n... and {row_count - len(rows[:10])} more rows")
            
            response_lines.append(f"\nTotal: {row_count} {'row' if row_count == 1 else 'rows'}")
    
    response = "\n".join(response_lines)
    
    # Build structured prompt monitor
    prompt_monitor = {
        "user_input": state.get("user_input", ""),
        "plan_explanation": state.get("plan_explain", ""),
        "execution_summary": {
            "query": state.get("plan", {}).get("sql", ""),
            "status": "success" if exec_result and not execution_stats.get("error") else "failed",
            "error": execution_stats.get("error"),
            "row_count": exec_result.get("row_count", 0),
            "column_count": len(columns),
            "duration_ms": execution_stats.get("execution_time_ms", 0)
        },
        "evaluation_notes": state.get("evaluator_notes", ""),
        "satisfaction": state.get("satisfaction", ""),
        "total_agent_duration_ms": total_ms,
        "clarify_turns": metrics.get("clarify_turns", 0),
        "full_logs": logs
    }
    
    logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"agent completed in {total_ms}ms"})
    
    final_output = {
        "response": response,
        "prompt_monitor": prompt_monitor,
    }
    
    # Update conversation history (short-term memory)
    conversation_history = state.get("conversation_history", [])
    conversation_entry = {
        "timestamp": _now_iso(),
        "query": state.get("user_input", ""),
        "plan_sql": state.get("plan", {}).get("sql", ""),
        "plan_quality": state.get("plan_quality", ""),
        "response": response,
        "satisfaction": state.get("satisfaction", ""),
        "row_count": exec_result.get("row_count", 0),
        "execution_error": execution_stats.get("error"),
        "prompt_monitor": prompt_monitor
    }
    conversation_history.append(conversation_entry)
    
    # Keep only last 5 interactions to prevent memory bloat
    conversation_history = conversation_history[-5:]
    
    return {
        "last_node": "end",
        "final_output": final_output,
        "conversation_history": conversation_history,
        "control": "end",
        "metrics": metrics,
        "logs": logs,
    }


