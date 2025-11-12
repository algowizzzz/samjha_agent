import json
import os
import glob
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from external.agent.schemas import AgentState, PartialState
from external.agent.config import QueryAgentConfig
from external.agent.state_manager import AgentStateManager
from tools.impl.nl_to_sql_planner import NLToSQLPlannerTool
from tools.impl.query_safety_validator import QuerySafetyValidatorTool
from tools.impl.query_result_evaluator import QueryResultEvaluatorTool
import hashlib

logger = logging.getLogger(__name__)

try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _make_node_result(
    state: AgentState,
    node_name: str,
    control: str,
    reasoning: str,
    node_output: Dict[str, Any],
    state_updates: Optional[Dict[str, Any]] = None
) -> PartialState:
    """
    Standardized node return format for consistent structure across all nodes.
    
    Args:
        state: Current agent state
        node_name: Name of the current node
        control: Next control signal
        reasoning: Human-readable explanation of the decision
        node_output: Node-specific output data
        state_updates: Additional state fields to update
    
    Returns:
        Partial state dict with standardized structure
    """
    logs = state.get("logs", [])
    logs.append({
        "node": node_name,
        "timestamp": _now_iso(),
        "msg": reasoning,
        "control": control
    })
    
    result = {
        "last_node": node_name,
                    "control": control,
        "node_reasoning": reasoning,
        "node_output": node_output,
        "logs": logs,
    }
    
    if state_updates:
        result.update(state_updates)
    
    return result


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
        # Check if custom data dictionary file is specified in state
        config_files = state.get("config_files", {})
        data_dict_file = config_files.get("data_dict_file")
        
        if data_dict_file:
            # Build path based on file name
            if os.path.exists(os.path.join("external", "config", "data_dictionary", data_dict_file)):
                data_dict_path = os.path.join("external", "config", "data_dictionary", data_dict_file)
            elif os.path.exists(os.path.join("external", "config", data_dict_file)):
                data_dict_path = os.path.join("external", "config", data_dict_file)
            else:
                data_dict_path = None
        else:
            # Default: Try risk dictionary first, fall back to standard dictionary
            data_dict_path = "external/config/data_dictionary/data_dictionary_risk.json"
            if not os.path.exists(data_dict_path):
                data_dict_path = "external/config/data_dictionary.json"
        
        data_dictionary = {}
        if data_dict_path and os.path.exists(data_dict_path):
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
    
    # Create human-friendly reasoning message
    if table_schema and len(table_schema) > 0:
        reasoning = "Data and chat history loaded"
    else:
        reasoning = "Initializing"
    
    return {
        "last_node": "invoke",
        "control": "check_followup",  # Check if this is a follow-up query before structure check
        "node_reasoning": reasoning,
        "parquet_location": parquet_location,
        "conversation_history": conversation_history,
        "table_schema": table_schema,
        "docs_meta": docs_meta,
        "logs": logs,
        "timestamp": _now_iso(),
        "clarification_count": 0,
        "sql_attempt_count": 0,
    }


def check_followup_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    """
    Check if the current query is a follow-up to a previous conversation.
    If conversation_history is not empty, this is a follow-up query.
    """
    conversation_history = state.get("conversation_history", [])
    conversation_history_raw = state.get("conversation_history_raw", [])
    
    # Check if either conversation history field has content
    is_followup = False
    if isinstance(conversation_history, list) and len(conversation_history) > 0:
        is_followup = True
    elif isinstance(conversation_history_raw, list) and len(conversation_history_raw) > 0:
        is_followup = True
    elif isinstance(conversation_history, str) and conversation_history != "No previous conversation history.":
        is_followup = True
    
    # Enhanced logging to show all conversation history sources
    conv_history_len = len(conversation_history) if isinstance(conversation_history, list) else 0
    conv_history_raw_len = len(conversation_history_raw) if isinstance(conversation_history_raw, list) else 0
    conv_history_str_len = len(conversation_history) if isinstance(conversation_history, str) else 0
    logger.info(f"[CHECK_FOLLOWUP] is_followup={is_followup}, conv_history(list)={conv_history_len}, conv_history_raw(list)={conv_history_raw_len}, conv_history(str)={conv_history_str_len}")
    
    # Route based on is_followup flag
    next_control = "check_data_sufficiency" if is_followup else "check_structure"
    
    # Human-friendly reasoning
    if is_followup:
        reasoning = "This appears to be a follow-up query in our ongoing conversation"
    else:
        reasoning = "Starting a new conversation"
    
    return _make_node_result(
        state=state,
        node_name="check_followup",
        control=next_control,
        reasoning=reasoning,
        node_output={
            "is_followup": is_followup,
            "conversation_history_length": len(conversation_history) if isinstance(conversation_history, list) else (len(conversation_history_raw) if isinstance(conversation_history_raw, list) else 0)
        },
        state_updates={
            "is_followup": is_followup,
            "more_data_needed": False if not is_followup else None  # Default to False for new queries
        }
    )


def check_data_sufficiency_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    """
    For follow-up queries, check if existing conversation history data is sufficient
    to answer the query, or if more data needs to be fetched.
    
    This node only runs when is_followup=True.
    """
    try:
        from external.agent.llm_client import get_llm_client
        llm_client = get_llm_client()
        
        if not llm_client.is_available():
            logger.warning("[CHECK_DATA_SUFFICIENCY] LLM not available, assuming more data needed")
            return _make_node_result(
                state=state,
                node_name="check_data_sufficiency",
                control="check_structure",
                reasoning="LLM not available, defaulting to more_data_needed=True",
                node_output={"more_data_needed": True},
                state_updates={"more_data_needed": True}
            )
        
        query = state.get("query", "")
        conversation_history = state.get("conversation_history", "No previous conversation history.")
        
        # Build prompt to assess data sufficiency
        system_prompt = """You are a data sufficiency analyzer. Your job is to determine if the PREVIOUS QUERY RESULTS contain enough data to answer the NEW FOLLOW-UP QUERY, or if a NEW database query is needed.

DECISION RULES:
1. If the new query asks to FILTER, COUNT, AGGREGATE, or ANALYZE data that was ALREADY RETURNED in previous results → more_data_needed=FALSE
2. If the new query asks for DIFFERENT data, DIFFERENT columns, or data NOT in previous results → more_data_needed=TRUE

Examples:
- Previous: "top 10 limits", New: "how many of these are in Americas?" → FALSE (data already there, just filter)
- Previous: "show Asia limits", New: "what are the specific IDs?" → FALSE (IDs already in previous results)
- Previous: "count by region", New: "show me the actual limit details" → TRUE (need detailed rows, not just counts)
- Previous: "top limits", New: "show me limits in breach" → TRUE (different filter criteria, need new query)

Respond in JSON format: {"more_data_needed": true/false, "reasoning": "explanation"}"""

        user_prompt = f"""Previous Conversation History:
{conversation_history}

New Follow-Up Query: {query}

Can this follow-up query be answered using ONLY the data from the previous query results, or do we need to fetch more/different data from the database?"""

        logger.info(f"[CHECK_DATA_SUFFICIENCY] Analyzing query: {query}")
        
        response = llm_client.invoke_with_prompt(system_prompt, user_prompt, response_format="json")
        response = response.strip()
        
        # Clean up JSON markers
        if response.startswith('```'):
            lines = response.split('\n')
            response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
        
        import json
        result = json.loads(response)
        more_data_needed = result.get("more_data_needed", True)
        reasoning = result.get("reasoning", "")
        
        logger.info(f"[CHECK_DATA_SUFFICIENCY] more_data_needed={more_data_needed}, reasoning={reasoning}")
        
        # Make reasoning more human-friendly
        if more_data_needed:
            friendly_reasoning = "This follow-up query requires different data than what we retrieved before - I'll fetch new data from the database"
        else:
            friendly_reasoning = "I can answer this using the data from our previous conversation"
        
        return _make_node_result(
            state=state,
            node_name="check_data_sufficiency",
            control="check_structure",
            reasoning=friendly_reasoning,
            node_output={
                "more_data_needed": more_data_needed,
                "data_sufficiency_reasoning": reasoning
            },
            state_updates={"more_data_needed": more_data_needed}
        )
        
    except Exception as e:
        logger.error(f"[CHECK_DATA_SUFFICIENCY] Error: {e}, defaulting to more_data_needed=True")
        return _make_node_result(
            state=state,
            node_name="check_data_sufficiency",
            control="check_structure",
            reasoning="I'll fetch new data to be safe",
            node_output={"more_data_needed": True},
            state_updates={"more_data_needed": True}
        )


def check_structure_node(state: AgentState, cfg: QueryAgentConfig, stream_callback: Optional[Callable[[str], None]] = None) -> PartialState:
    """
    Check if query is structured (data retrieval) or unstructured (knowledge question).
    Uses LLM to classify query intent.
    """
    try:
        from external.agent.llm_client import get_llm_client
        llm_client = get_llm_client()
        
        if not llm_client.is_available():
            # Default to structured if LLM unavailable
            return _make_node_result(
                state=state,
                node_name="check_structure",
                control="check_ambiguity",
                reasoning="LLM unavailable, assuming structured query",
                node_output={"is_structured": True, "confidence": "low"},
                state_updates={"is_structured": True}
            )
        
        # Build prompt for structure classification
        system_prompt = cfg.get_nested("prompts", "check_structure_system", default="""
You are a query classifier. Determine if the user's query is:
- STRUCTURED: Asking for data retrieval from database (show me, get, list, top N, etc.)
- UNSTRUCTURED: Asking for knowledge/definitions (what is, define, explain, what does mean, etc.)

Respond with JSON:
{
  "is_structured": true/false,
  "confidence": "high|medium|low",
  "reasoning": "Brief explanation"
}
""")
        
        # Format schema summary
        table_schema = state.get("table_schema", {})
        schema_lines = []
        for table, schema_info in table_schema.items():
            cols = schema_info.get("columns", [])
            col_names = [c.get("name", "") for c in cols if isinstance(c, dict)]
            if col_names:
                schema_lines.append(f"- {table}: {', '.join(col_names[:5])}")
        schema_summary = "\n".join(schema_lines) if schema_lines else "No tables available"
        
        # Format business context
        docs_meta = state.get("docs_meta", [])
        business_lines = []
        for item in docs_meta:
            if item.get("type") == "business_glossary":
                glossary = item.get("glossary", {})
                business_lines.append("Business Terms: " + ", ".join(list(glossary.keys())[:10]))
        business_context = "\n".join(business_lines) if business_lines else ""
        
        user_prompt = f"""User Query: {state.get('user_input', '')}

Available Tables:
{schema_summary}

{business_context}

Classify if this is a STRUCTURED data query or UNSTRUCTURED knowledge question."""
        
        # Call LLM
        response = llm_client.invoke_with_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        )
        
        # Parse response
        import json
        try:
            result = json.loads(response.strip())
            is_structured = result.get("is_structured", True)
            confidence = result.get("confidence", "medium")
            reasoning = result.get("reasoning", "Classification completed")
        except:
            # Fallback: assume structured
            is_structured = True
            confidence = "low"
            reasoning = "Failed to parse LLM response, defaulting to structured"
        
        # Determine next control
        if is_structured:
            next_control = "check_ambiguity"
        else:
            next_control = "end"  # Knowledge questions go to end_node for glossary lookup
        
        # Human-friendly reasoning
        if is_structured:
            friendly_reasoning = "This looks like a data query - I'll search the database"
        else:
            friendly_reasoning = "This looks like a knowledge question - I'll check the documentation"
        
        return _make_node_result(
            state=state,
            node_name="check_structure",
            control=next_control,
            reasoning=friendly_reasoning,
            node_output={
                "is_structured": is_structured,
                "confidence": confidence,
                "classification_reasoning": reasoning
            },
            state_updates={"is_structured": is_structured}
        )
        
    except Exception as e:
        # On error, default to structured and continue
        return _make_node_result(
            state=state,
            node_name="check_structure",
            control="check_ambiguity",
            reasoning="I'll search the database for this",
            node_output={"is_structured": True, "error": str(e)},
            state_updates={"is_structured": True}
        )


def check_ambiguity_node(state: AgentState, cfg: QueryAgentConfig, stream_callback: Optional[Callable[[str], None]] = None) -> PartialState:
    """
    Check if query has clear intent and sufficient details.
    Uses clarification gate logic to determine if questions are needed.
    """
    try:
        # Check clarification count for retry limit
        clarification_count = state.get("clarification_count", 0)
        max_clarify_turns = cfg.get_nested("limits", "max_clarify_turns", default=3)
        
        if clarification_count >= max_clarify_turns:
            # Hit max retries, proceed anyway
            return _make_node_result(
                state=state,
                node_name="check_ambiguity",
                control="generate_sql",
                reasoning=f"Max clarification attempts ({max_clarify_turns}) reached, proceeding with best effort",
                node_output={"is_ambiguous": False, "forced": True},
                state_updates={"is_ambiguous": False}
            )
        
        # Use NLToSQLPlannerTool's clarification gate
        planner = NLToSQLPlannerTool({
            "preview_rows": cfg.get_nested("duckdb", "preview_rows", default=100),
            "data_directory": state.get("parquet_location", "data/duckdb"),
            "use_llm": True,
        })
        
        # Build query context (include accumulated clarifications if any)
        # If clarification is a complete query (starts with query verbs), use it as primary query
        accumulated_clarifications = state.get("accumulated_clarifications", [])
        original_query = state.get("user_input", "")
        
        # Check if latest clarification is a complete query
        if accumulated_clarifications:
            latest_clarification = accumulated_clarifications[-1].strip()
            # Detect if clarification is a complete query (not just an answer)
            query_verbs = ["show", "get", "list", "find", "select", "count", "sum", "display", "return"]
            is_complete_query = any(latest_clarification.lower().startswith(verb) for verb in query_verbs)
            
            if is_complete_query:
                # Use clarification as primary query, original as context
                query = latest_clarification
                if len(accumulated_clarifications) > 1:
                    # Include previous clarifications as context
                    query += f"\n\nOriginal query: {original_query}"
                    for i, clarif in enumerate(accumulated_clarifications[:-1], 1):
                        query += f"\n\nPrevious clarification {i}: {clarif}"
            else:
                # Clarification is just an answer, append to original query
                query = original_query
                for i, clarif in enumerate(accumulated_clarifications, 1):
                    query += f"\n\nClarification {i}: {clarif}"
        else:
            query = original_query
        
        # Get previous clarification questions (for context)
        previous_clarifications = state.get("clarification_questions", [])
        
        # Get follow-up flags
        is_followup = state.get("is_followup", False)
        more_data_needed = state.get("more_data_needed", False)
        logger.info(f"[CHECK_AMBIGUITY] is_followup={is_followup}, more_data_needed={more_data_needed}, passing to planner")
        
        # Execute planner (includes clarification gate)
        # Get config file from state
        config_files = state.get("config_files", {})
        data_dict_file = config_files.get("data_dict_file")
        
        res = planner.execute({
            "query": query,
            "table_schema": state.get("table_schema", {}),
            "docs_meta": state.get("docs_meta", []),
            "previous_clarifications": previous_clarifications,
            "conversation_history": state.get("conversation_history", "No previous conversation history."),
            "is_followup": is_followup,
            "more_data_needed": more_data_needed,
            "data_dict_file": data_dict_file,
        }, stream_callback=stream_callback)
        
        # Check if clarification is needed
        if res.get("type") == "clarification_needed":
            clarification_questions = res.get("clarification_questions", [])
            plan_explain = res.get("plan_explain", "")
            
            return _make_node_result(
                state=state,
                node_name="check_ambiguity",
                control="clarify",
                reasoning=f"Query is ambiguous, need clarification ({len(clarification_questions)} questions)",
                node_output={
                    "is_ambiguous": True,
                "clarification_questions": clarification_questions,
                    "plan_explain": plan_explain
                },
                state_updates={
                    "is_ambiguous": True,
                    "clarification_questions": clarification_questions,
                    "plan_explain": plan_explain,
                    "ambiguity_reasons": [plan_explain]
                }
            )
        else:
            # Query is clear, proceed to SQL generation
            plan = res.get("plan", {})
        plan_quality = res.get("plan_quality", "high")
        plan_explain = res.get("plan_explain", "")
        
        # Human-friendly reasoning
        friendly_reasoning = "Your question is clear - I'll generate the SQL query now"
        
        return _make_node_result(
            state=state,
            node_name="check_ambiguity",
                control="generate_sql",
                reasoning=friendly_reasoning,
                node_output={
                    "is_ambiguous": False,
            "plan": plan,
            "plan_quality": plan_quality,
                    "plan_explain": plan_explain
                },
                state_updates={
                    "is_ambiguous": False,
                    "plan": plan,
                    "plan_quality": plan_quality,
                    "plan_explain": plan_explain
                }
            )
    
    except Exception as e:
        # On error, assume not ambiguous and continue
        return _make_node_result(
            state=state,
            node_name="check_ambiguity",
            control="generate_sql",
            reasoning=f"Ambiguity check failed: {str(e)}, proceeding with SQL generation",
            node_output={"is_ambiguous": False, "error": str(e)},
            state_updates={"is_ambiguous": False}
        )


def process_clarification_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    """
    Process user's clarification response and prepare for re-check.
    Merges clarification with original query context.
    """
    user_clarification = state.get("user_clarification", "")
    clarification_count = state.get("clarification_count", 0) + 1
    
    # Accumulate clarifications
    accumulated_clarifications = state.get("accumulated_clarifications", [])
    accumulated_clarifications.append(user_clarification)
    
    reasoning = f"Processed clarification #{clarification_count}: {user_clarification[:50]}..."
    
    return _make_node_result(
        state=state,
        node_name="process_clarification",
        control="check_ambiguity",  # Re-check ambiguity with new info
        reasoning=reasoning,
        node_output={
            "clarification_processed": user_clarification,
            "total_clarifications": clarification_count
        },
        state_updates={
            "clarification_count": clarification_count,
            "accumulated_clarifications": accumulated_clarifications
        }
    )


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
        
        # Get clarify template from config
        clarify_template = cfg.get_nested("prompts", "clarify_template", default="""I need clarification on your query: "{user_input}"

**Why I'm asking:**
{reasoning_text}

**Questions:**
{questions_text}

Please provide more details to help me generate an accurate query.""")
        
        clarify_prompt = clarify_template.format(
            user_input=user_input,
            reasoning_text=reasoning_text,
            questions_text=questions_text
        )
        
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


def generate_sql_node(state: AgentState, cfg: QueryAgentConfig, stream_callback: Optional[Callable[[str], None]] = None) -> PartialState:
    """
    Generate SQL query using NLToSQLPlannerTool.
    This node is called after ambiguity check confirms query is clear.
    """
    try:
        # Check if plan already exists from check_ambiguity_node
        plan = state.get("plan")
        if plan and plan.get("sql"):
            # Plan already generated, just proceed
            sql_attempt_count = state.get("sql_attempt_count", 0) + 1
            return _make_node_result(
                state=state,
                node_name="generate_sql",
                control="execute_sql",
                reasoning=f"Using pre-generated SQL from ambiguity check (attempt #{sql_attempt_count})",
                node_output={
                    "sql": plan.get("sql"),
                    "plan_quality": state.get("plan_quality", "high"),
                    "plan_explain": state.get("plan_explain", "")
                },
                state_updates={"sql_attempt_count": sql_attempt_count}
            )
        
        # Otherwise, generate SQL fresh
        planner = NLToSQLPlannerTool({
            "preview_rows": cfg.get_nested("duckdb", "preview_rows", default=100),
            "data_directory": state.get("parquet_location", "data/duckdb"),
            "use_llm": True,
        })
        
        # Build query context (same logic as check_ambiguity_node)
        accumulated_clarifications = state.get("accumulated_clarifications", [])
        original_query = state.get("user_input", "")
        
        # Check if latest clarification is a complete query
        if accumulated_clarifications:
            latest_clarification = accumulated_clarifications[-1].strip()
            query_verbs = ["show", "get", "list", "find", "select", "count", "sum", "display", "return"]
            is_complete_query = any(latest_clarification.lower().startswith(verb) for verb in query_verbs)
            
            if is_complete_query:
                query = latest_clarification
                if len(accumulated_clarifications) > 1:
                    query += f"\n\nOriginal query: {original_query}"
                    for i, clarif in enumerate(accumulated_clarifications[:-1], 1):
                        query += f"\n\nPrevious clarification {i}: {clarif}"
            else:
                query = original_query
                for i, clarif in enumerate(accumulated_clarifications, 1):
                    query += f"\n\nClarification {i}: {clarif}"
        else:
            query = original_query
        
        # Add error context if this is a retry
        execution_error = state.get("execution_stats", {}).get("error")
        if execution_error:
            query += f"\n\nPrevious SQL failed with error: {execution_error}"
            query += "\nPlease generate corrected SQL."
        
        # Get follow-up flags
        is_followup = state.get("is_followup", False)
        more_data_needed = state.get("more_data_needed", False)
        logger.info(f"[GENERATE_SQL] is_followup={is_followup}, more_data_needed={more_data_needed}, passing to planner")
        
        # Get config file from state
        config_files = state.get("config_files", {})
        data_dict_file = config_files.get("data_dict_file")
        
        # Execute planner
        res = planner.execute({
            "query": query,
            "table_schema": state.get("table_schema", {}),
            "docs_meta": state.get("docs_meta", []),
            "previous_clarifications": state.get("clarification_questions", []),
            "conversation_history": state.get("conversation_history", "No previous conversation history."),
            "is_followup": is_followup,
            "more_data_needed": more_data_needed,
            "data_dict_file": data_dict_file,
        }, stream_callback=stream_callback)
        
        plan = res.get("plan", {})
        plan_quality = res.get("plan_quality", "medium")
        plan_explain = res.get("plan_explain", "")
        sql_attempt_count = state.get("sql_attempt_count", 0) + 1
        
        # Log the SQL that was generated by the planner
        generated_sql = plan.get("sql", "")
        logger.info(f"[GENERATE_SQL] Planner generated SQL: {generated_sql[:200] if generated_sql else 'EMPTY'}...")
        logger.info(f"[GENERATE_SQL] Full SQL length: {len(generated_sql)} chars")
        logger.info(f"[GENERATE_SQL] Full SQL: {generated_sql}")
        logger.info(f"[GENERATE_SQL] Plan object keys: {list(plan.keys())}")
        logger.info(f"[GENERATE_SQL] Plan object: {plan}")
        
        # Human-friendly reasoning
        friendly_reasoning = "SQL query generated successfully"
        
        return _make_node_result(
            state=state,
            node_name="generate_sql",
            control="execute_sql",
            reasoning=friendly_reasoning,
            node_output={
                "sql": plan.get("sql", ""),
                "plan_quality": plan_quality,
                "plan_explain": plan_explain
            },
            state_updates={
            "plan": plan,
            "plan_quality": plan_quality,
                "plan_explain": plan_explain,
                "sql_attempt_count": sql_attempt_count
            }
        )
    
    except Exception as e:
        return _make_node_result(
            state=state,
            node_name="generate_sql",
            control="end",
            reasoning="I had trouble generating the SQL query",
            node_output={"error": str(e)},
            state_updates={"plan": None, "plan_quality": "error", "plan_explain": f"Error: {str(e)}"}
        )


def execute_sql_node(state: AgentState, cfg: QueryAgentConfig) -> PartialState:
    """
    Execute SQL query using DuckDB.
    Routes to retry_sql on error or synthesize on success.
    """
    logs = state.get("logs", [])
    logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": "executing query"})
    
    # Check if this is a knowledge question (no SQL execution needed)
    plan_sql = (state.get("plan") or {}).get("sql", "SELECT 1")
    logger.info(f"[EXECUTE_SQL] Retrieved plan_sql from state: {plan_sql[:200] if plan_sql else 'EMPTY'}...")
    logger.info(f"[EXECUTE_SQL] Full plan_sql length: {len(plan_sql)} chars")
    logger.info(f"[EXECUTE_SQL] Full plan_sql: {plan_sql}")
    if plan_sql.startswith("-- KNOWLEDGE QUESTION"):
        logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": "knowledge question detected, skipping SQL execution"})
        # Return the explanation as the result
        explanation = state.get("plan_explain", "")
        if not explanation:
            explanation = "I detected this as a knowledge question, but no explanation was provided in the plan. Please check the business glossary or data dictionary."
        logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": f"knowledge response: {explanation[:100]}..."})
        return _make_node_result(
            state=state,
            node_name="execute_sql",
            control="synthesize",  # Go to synthesize for knowledge response
            reasoning="This is a knowledge question - checking documentation instead",
            node_output={
            "execution_result": {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "query": plan_sql
            },
            "execution_stats": {
                "execution_time_ms": 0,
                "error": None,
                "limited": False,
                "knowledge_response": explanation
                }
            },
            state_updates={
                "execution_result": {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "query": plan_sql
                },
                "execution_stats": {
                    "execution_time_ms": 0,
                    "error": None,
                    "limited": False,
                    "knowledge_response": explanation
                }
            }
        )
    
    if duckdb is None:
        logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": "DuckDB not installed", "level": "error"})
        return _make_node_result(
            state=state,
            node_name="execute_sql",
            control="end",
            reasoning="DuckDB not installed, cannot execute query",
            node_output={"error": "DuckDB not installed"},
            state_updates={
            "execution_result": None,
                "execution_stats": {"error": "DuckDB not installed", "execution_time_ms": 0}
        }
        )
    
    try:
        safety = QuerySafetyValidatorTool({
            "limits": cfg.get("limits") or {"max_rows": 1000},
            "safety": cfg.get("safety") or {},
        })
        preview_rows = int(cfg.get_nested("duckdb", "preview_rows", default=100))
        
        logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": f"validating query: {plan_sql[:50]}..."})
        logger.info(f"[EXECUTE_SQL] Plan SQL before safety validation: {plan_sql[:200] if plan_sql else 'EMPTY'}...")
        
        safe = safety.execute({
            "query": plan_sql,
            "enforce_limit": True,
            "default_limit": preview_rows,
        })
        sanitized_sql = safe.get("sanitized_query", plan_sql)
        logger.info(f"[EXECUTE_SQL] Safety tool sanitized SQL: {sanitized_sql[:200] if sanitized_sql else 'EMPTY'}...")
        logger.info(f"[EXECUTE_SQL] SQL changed by safety tool: {plan_sql != sanitized_sql}")
        if not safe.get("is_safe", False):
            error_msg = safe.get("reason", "unsafe query")
            logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": f"query validation failed: {error_msg}", "level": "error"})
            
            # Check if we can retry
            sql_attempt_count = state.get("sql_attempt_count", 0)
            max_sql_attempts = cfg.get_nested("limits", "max_sql_attempts", default=2)
            
            if sql_attempt_count < max_sql_attempts:
                next_control = "retry_sql"
                reasoning = f"Query validation failed: {error_msg}, will retry ({sql_attempt_count}/{max_sql_attempts})"
            else:
                next_control = "end"
                reasoning = f"Query validation failed: {error_msg}, max attempts reached"
            
            return _make_node_result(
                state=state,
                node_name="execute_sql",
                control=next_control,
                reasoning=reasoning,
                node_output={"error": error_msg, "validation_failed": True},
                state_updates={
                "execution_result": None,
                    "execution_stats": {"error": error_msg, "execution_time_ms": 0, "limited": False}
            }
            )
        sql = safe.get("sanitized_query", plan_sql)
        
        logger.info(f"[EXECUTE_SQL] Final SQL to execute: {sql[:200] if sql else 'EMPTY'}...")
        logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": "query validated, executing..."})
        
        # Execute query directly with DuckDB (standalone mode)
        # Get data directory from config or env
        import os
        data_dir = state.get('parquet_location', os.getenv('DUCKDB_DATA_DIR', 'data/duckdb'))
        
        start_time = time.time()
        # Use in-memory connection (DuckDB loads files directly)
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
        
        # Format result and convert dates to strings for JSON serialization
        def convert_value(val):
            """Convert date/datetime objects to ISO format strings"""
            if hasattr(val, 'isoformat'):
                return val.isoformat()
            return val
        
        rows_with_converted_dates = [
            {col: convert_value(val) for col, val in zip(columns, row)}
            for row in result
        ]
        
        result_payload = {
            "query": sql,
            "columns": columns,
            "rows": rows_with_converted_dates,
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
        
        logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": f"query executed successfully: {result_payload['row_count']} rows returned"})
        
        # Human-friendly reasoning
        row_count = result_payload['row_count']
        if row_count == 0:
            friendly_reasoning = "Query executed - no results found"
        elif row_count == 1:
            friendly_reasoning = "Query executed - found 1 result"
        else:
            friendly_reasoning = f"Query executed - found {row_count} results"
        
        return _make_node_result(
            state=state,
            node_name="execute_sql",
            control="synthesize",  # Skip evaluation, go straight to response synthesis
            reasoning=friendly_reasoning,
            node_output={
                "execution_result": result_payload,
                "execution_stats": stats
            },
            state_updates={
            "execution_result": result_payload,
            "execution_stats": stats,
                "metrics": metrics
        }
        )
    except Exception as e:
        error_msg = str(e)
        logs.append({"node": "execute_sql", "timestamp": _now_iso(), "msg": f"query execution failed: {error_msg}", "level": "error"})
        
        # Check if we can retry
        sql_attempt_count = state.get("sql_attempt_count", 0)
        max_sql_attempts = cfg.get_nested("limits", "max_sql_attempts", default=2)
        
        if sql_attempt_count < max_sql_attempts:
            next_control = "retry_sql"
            reasoning = f"Query execution failed: {error_msg}, will retry ({sql_attempt_count}/{max_sql_attempts})"
        else:
            next_control = "end"
            reasoning = f"Query execution failed: {error_msg}, max attempts reached"
        
        return _make_node_result(
            state=state,
            node_name="execute_sql",
            control=next_control,
            reasoning=reasoning,
            node_output={"error": error_msg, "execution_failed": True},
            state_updates={
            "execution_result": None,
                "execution_stats": {"error": error_msg, "execution_time_ms": 0, "limited": False}
            }
        )


def retry_sql_node(state: AgentState, cfg: QueryAgentConfig, stream_callback: Optional[Callable[[str], None]] = None) -> PartialState:
    """
    Retry SQL generation after execution error.
    Passes error context to SQL generator for correction.
    """
    sql_attempt_count = state.get("sql_attempt_count", 0)
    max_sql_attempts = cfg.get_nested("limits", "max_sql_attempts", default=2)
    
    if sql_attempt_count >= max_sql_attempts:
        # Max attempts reached, give up
        return _make_node_result(
            state=state,
            node_name="retry_sql",
            control="end",
            reasoning=f"Max SQL attempts ({max_sql_attempts}) reached, giving up",
            node_output={"max_attempts_reached": True},
            state_updates={}
        )
    
    # Reset plan so generate_sql_node will regenerate
    return _make_node_result(
        state=state,
        node_name="retry_sql",
        control="generate_sql",  # Go back to SQL generation with error context
        reasoning=f"Retrying SQL generation (attempt {sql_attempt_count + 1}/{max_sql_attempts})",
        node_output={"retry_triggered": True, "attempt": sql_attempt_count + 1},
        state_updates={
            "plan": None  # Clear plan to force regeneration
        }
    )


def end_node(state: AgentState, cfg: QueryAgentConfig, stream_callback_response: Optional[Callable[[str], None]] = None, stream_callback_prompt_monitor: Optional[Callable[[str], None]] = None) -> PartialState:
    """
    Finalize agent response and generate insights.
    NOTE: In new workflow, this serves as the synthesize_response_node.
    """
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
    
    # Get execution results
    exec_result = state.get("execution_result") or {}
    rows = exec_result.get("rows", [])
    columns = exec_result.get("columns", [])
    execution_stats = state.get("execution_stats", {})
    user_query = state.get("user_input", "")
    plan = state.get("plan") or {}
    plan_sql = plan.get("sql", "") if plan else ""
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[END_NODE] Execution result check: row_count={exec_result.get('row_count', 0)}, columns={len(columns)}, rows={len(rows)}, sql={plan_sql[:100] if plan_sql else 'N/A'}")
    logger.info(f"[END_NODE] execution_result type: {type(exec_result)}, has 'rows' key: {'rows' in exec_result}, has 'columns' key: {'columns' in exec_result}")
    if exec_result:
        logger.info(f"[END_NODE] execution_result keys: {list(exec_result.keys())}")
        logger.info(f"[END_NODE] execution_result.query: {exec_result.get('query', 'MISSING')[:100] if exec_result.get('query') else 'MISSING'}")
    
    # CRITICAL: Check if state already has a raw_table that might be from a previous query
    existing_raw_table = state.get("raw_table", {})
    if existing_raw_table:
        logger.warning(f"[END_NODE] State has existing raw_table! Query: '{existing_raw_table.get('query', 'N/A')[:100]}', row_count: {existing_raw_table.get('row_count', 0)}")
        if existing_raw_table.get("query") != plan_sql:
            logger.warning(f"[END_NODE] Existing raw_table query doesn't match current plan_sql! Will be overwritten.")
    
    # Check if this is a knowledge question response
    knowledge_response = execution_stats.get("knowledge_response")
    if knowledge_response:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[END_NODE] Knowledge response detected: '{knowledge_response[:100]}...'")
        logs.append({"node": "end", "timestamp": _now_iso(), "msg": "returning knowledge response (no LLM generation needed)"})
        
        result = {
            "last_node": "end",
            "control": "end",
            "final_output": {
                "raw_table": {
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "query": plan_sql
                },
                "response": knowledge_response,
                "prompt_monitor": {
                    "procedural_reasoning": "Knowledge question answered from business glossary - no data query executed.",
                    "raw_state": {
                        "user_input": user_query,
                        "plan_explanation": knowledge_response,
                        "type": "knowledge_response"
                    }
                }
            },
            "metrics": metrics,
            "logs": logs,
        }
        logger.info(f"[END_NODE] Returning knowledge result with response length: {len(knowledge_response)}")
        return result
    
    # ============================================================================
    # OUTPUT 1: Raw Table Data
    # ============================================================================
    # CRITICAL: For conversation history, use plan_sql (the original LLM-generated SQL)
    # because follow-up queries need to filter based on the original query structure.
    # The sanitized_query from exec_result might be simplified/modified by the safety tool.
    # However, for raw_table display, we can show the executed SQL.
    executed_sql = exec_result.get("query", plan_sql)  # The SQL that was actually executed (may be sanitized)
    actual_sql = plan_sql  # Use original planned SQL for conversation history (what LLM generated)
    raw_table = {
        "columns": columns,
        "rows": rows,
        "row_count": exec_result.get("row_count", 0),
        "query": executed_sql  # Show executed SQL in raw_table (for debugging/display)
    }
    
    # Debug: Log raw_table structure
    logger.info(f"[END_NODE] Raw table created: row_count={raw_table['row_count']}, columns={len(raw_table['columns'])}, rows={len(raw_table['rows'])}, query_length={len(plan_sql) if plan_sql else 0}")
    
    # ============================================================================
    # OUTPUT 2: LLM-Generated Response/Insights
    # ============================================================================
    llm_response = None
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        from external.agent.llm_client import get_llm_client
        llm_client = get_llm_client()
        
        # DEBUG: Check LLM availability
        is_available = llm_client.is_available()
        logger.info(f"[END_NODE] LLM client available: {is_available}")
        logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM client available: {is_available}"})
        
        if is_available:
            logs.append({"node": "end", "timestamp": _now_iso(), "msg": "generating LLM response/insights"})
            
            # Build system prompt from config
            system_prompt = cfg.get_nested("prompts", "end_response_system", default="")
            logger.info(f"[END_NODE] Config prompt length: {len(system_prompt) if system_prompt else 0}")
            logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"Config prompt loaded: {len(system_prompt) if system_prompt else 0} chars"})
            
            if not system_prompt:
                system_prompt = """You are a data analyst assistant. Generate insights and a user-friendly response based on the query results.

IMPORTANT: The raw table data and prompt monitor are provided separately. Focus on generating insights, patterns, and actionable observations from the data."""
                logger.info("[END_NODE] Using default system prompt (config was empty)")
            
            # Load procedural knowledge from data_dictionary
            procedural_knowledge = ""
            try:
                import os
                # Check if custom data dictionary file is specified in state
                config_files = state.get("config_files", {})
                data_dict_file = config_files.get("data_dict_file")
                
                if data_dict_file:
                    # Build path based on file name
                    if os.path.exists(os.path.join("external", "config", "data_dictionary", data_dict_file)):
                        data_dict_path = os.path.join("external", "config", "data_dictionary", data_dict_file)
                    elif os.path.exists(os.path.join("external", "config", data_dict_file)):
                        data_dict_path = os.path.join("external", "config", data_dict_file)
                    else:
                        data_dict_path = None
                else:
                    # Default: Try risk dictionary first, fall back to standard dictionary
                    data_dict_path = "external/config/data_dictionary/data_dictionary_risk.json"
                    if not os.path.exists(data_dict_path):
                        data_dict_path = "external/config/data_dictionary.json"
                
                if data_dict_path and os.path.exists(data_dict_path):
                    with open(data_dict_path, 'r', encoding='utf-8') as f:
                        data_dict = json.load(f)
                        procedural_knowledge = data_dict.get("procedural_knowledge", "")
            except Exception:
                pass  # Optional, graceful failure
            
            # Build user prompt with full state (JSON serialized)
            # Limit rows to first 50 to avoid token overflow
            # Send ALL rows to LLM for comprehensive analysis (not just sample)
            all_rows = rows if rows else []
            state_summary = {
                "user_query": user_query,
                "sql_query": plan_sql,
                "execution_status": "success" if not execution_stats.get("error") else f"failed: {execution_stats.get('error')}",
                "row_count": exec_result.get("row_count", 0),
                "columns": columns,
                "all_rows": all_rows,  # Changed from sample_rows to all_rows
                "plan_quality": state.get("plan_quality", ""),
                "plan_explanation": state.get("plan_explain", ""),
                "evaluation_notes": state.get("evaluator_notes", ""),
                "satisfaction": state.get("satisfaction", ""),
                "docs_meta": state.get("docs_meta", []),  # Data dictionary
                "table_schema": state.get("table_schema", {}),
                "procedural_knowledge": procedural_knowledge
            }
            
            # Get user prompt template from config
            user_template = cfg.get_nested("prompts", "end_response_user_template", default="""Generate a user-friendly response with insights from this query execution.

Full State Summary:
{state_summary}

Use the data dictionary (docs_meta) to understand column meanings and business context. Focus on what the data tells us, key patterns, and actionable observations.""")
            
            # Format results summary - include ALL rows for comprehensive analysis
            results_summary = {
                "row_count": exec_result.get("row_count", 0),
                "columns": columns,
                "all_rows": all_rows,  # Changed from sample_rows to all_rows - send all data for analysis
                "execution_status": "success" if not execution_stats.get("error") else f"failed: {execution_stats.get('error')}",
                "execution_time_ms": execution_stats.get("execution_time_ms", 0)
            }
            
            # Use standard formatted conversation history (same format as check_ambiguity_node and generate_sql_node)
            conversation_history_formatted = state.get("conversation_history", "No previous conversation history.")
            
            # Build clarification history formatted string
            accumulated_clarifications = state.get("accumulated_clarifications", [])
            clarification_questions = state.get("clarification_questions", [])
            clarification_history_formatted = "No clarifications were needed for this query."
            if accumulated_clarifications or clarification_questions:
                clarif_lines = []
                if clarification_questions:
                    clarif_lines.append("Clarification Questions Asked:")
                    for i, q in enumerate(clarification_questions, 1):
                        clarif_lines.append(f"  {i}. {q}")
                if accumulated_clarifications:
                    clarif_lines.append("\nUser Clarifications Provided:")
                    for i, clarif in enumerate(accumulated_clarifications, 1):
                        clarif_lines.append(f"  {i}. {clarif}")
                if clarif_lines:
                    clarification_history_formatted = "\n".join(clarif_lines)
            
            # Extract business glossary from docs_meta
            business_glossary_formatted = "No business glossary available."
            docs_meta = state.get("docs_meta", [])
            for item in docs_meta:
                if item.get("type") == "business_glossary":
                    glossary = item.get("glossary", {})
                    if glossary:
                        gloss_lines = []
                        for term, definition in glossary.items():
                            if isinstance(definition, dict):
                                desc = definition.get("definition", definition.get("description", ""))
                                usage = definition.get("usage", definition.get("context", ""))
                                gloss_lines.append(f"**{term}**: {desc}")
                                if usage:
                                    gloss_lines.append(f"  Usage: {usage}")
                            else:
                                gloss_lines.append(f"**{term}**: {definition}")
                        if gloss_lines:
                            business_glossary_formatted = "\n".join(gloss_lines)
                    break
            
            user_prompt = user_template.format(
                user_query=user_query,
                sql_query=plan_sql,
                results_summary=json.dumps(results_summary, indent=2),
                conversation_history=conversation_history_formatted,
                clarification_history=clarification_history_formatted,
                business_glossary=business_glossary_formatted,
                state_summary=json.dumps(state_summary, indent=2)  # Keep for backward compatibility
            )
            
            logger.info(f"[END_NODE] Calling LLM with prompt length: {len(user_prompt)} chars")
            try:
                # Use streaming if callback provided, otherwise use regular invoke
                if stream_callback_response:
                    full_response = ""
                    for chunk in llm_client.stream_with_prompt(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.3,  # Slightly higher for more creative insights
                        callback=stream_callback_response
                    ):
                        full_response += chunk
                    llm_response = full_response
                else:
                    llm_response = llm_client.invoke_with_prompt(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.3  # Slightly higher for more creative insights
                    )
                logger.info(f"[END_NODE] LLM response received: {len(llm_response) if llm_response else 0} chars")
                logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM response generated successfully ({len(llm_response) if llm_response else 0} chars)"})
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"[END_NODE] LLM response generation failed: {str(e)}\n{error_trace}")
                logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM response generation failed: {str(e)}", "level": "error", "traceback": error_trace})
                llm_response = None
        else:
            logger.warning("[END_NODE] LLM not available, using template response")
            logs.append({"node": "end", "timestamp": _now_iso(), "msg": "LLM not available, using template response"})
    except Exception as e:
        import traceback
        import logging
        logger = logging.getLogger(__name__)
        error_trace = traceback.format_exc()
        logger.error(f"[END_NODE] LLM client error: {str(e)}\n{error_trace}")
        logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM client error: {str(e)}", "level": "error", "traceback": error_trace})
        llm_response = None
    
    # Fallback to template if LLM failed
    if not llm_response:
        response_lines = []
        if execution_stats.get("error"):
            response_lines.append(f"❌ Query failed: {execution_stats['error']}")
            response_lines.append(f"\nQuery attempted: {plan_sql}")
        else:
            row_count = exec_result.get("row_count", 0)
            if row_count == 0:
                response_lines.append(f"No results found for: {user_query}")
            else:
                response_lines.append(f"Here are the results for '{user_query}':\n")
                preview_rows_list = rows[:10]
                for i, row in enumerate(preview_rows_list, 1):
                    row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                    response_lines.append(f"{i}. {row_str}")
                if row_count > 10:
                    response_lines.append(f"\n... and {row_count - 10} more rows")
                response_lines.append(f"\nTotal: {row_count} {'row' if row_count == 1 else 'rows'}")
        llm_response = "\n".join(response_lines)
    
    # ============================================================================
    # OUTPUT 3: LLM-Generated Prompt Monitor (Procedural Reasoning)
    # ============================================================================
    llm_prompt_monitor = None
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        from external.agent.llm_client import get_llm_client
        llm_client = get_llm_client()
        
        # DEBUG: Check LLM availability
        is_available = llm_client.is_available()
        logger.info(f"[END_NODE] LLM client available for prompt monitor: {is_available}")
        logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM client available for prompt monitor: {is_available}"})
        
        if is_available:
            logs.append({"node": "end", "timestamp": _now_iso(), "msg": "generating LLM prompt monitor/reasoning"})
            
            # Build system prompt from config
            system_prompt = cfg.get_nested("prompts", "end_prompt_monitor_system", default="")
            logger.info(f"[END_NODE] Config prompt_monitor length: {len(system_prompt) if system_prompt else 0}")
            logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"Config prompt_monitor loaded: {len(system_prompt) if system_prompt else 0} chars"})
            
            if not system_prompt:
                system_prompt = """You are documenting the agent's reasoning process. Generate a procedural summary of what was done and why.

Focus on:
- Why the query was structured this way
- What decisions were made during planning
- How the execution succeeded or failed
- What the evaluation determined
- The overall reasoning chain

Format as a clear, structured procedural summary, not a response to the user."""
                logger.info("[END_NODE] Using default prompt_monitor system prompt (config was empty)")
            
            # Extract conversation history for prompt monitor
            conversation_history_raw = state.get("conversation_history_raw", [])
            conversation_history_formatted = []
            if conversation_history_raw:
                for i, turn in enumerate(conversation_history_raw, 1):
                    conv_turn = {
                        "turn_number": i,
                        "user_query": turn.get("query", ""),
                        "sql_executed": turn.get("sql", "") if turn.get("sql") and turn.get("sql") != "-- KNOWLEDGE QUESTION" else None,
                        "response_summary": turn.get("response", "")[:200] + "..." if len(turn.get("response", "")) > 200 else turn.get("response", ""),
                        "row_count": turn.get("row_count", 0)
                    }
                    # Include prompt monitor if available
                    prompt_monitor = turn.get('prompt_monitor')
                    if prompt_monitor:
                        if isinstance(prompt_monitor, dict):
                            reasoning = prompt_monitor.get('procedural_reasoning', '') or prompt_monitor.get('reasoning', '')
                            if reasoning:
                                conv_turn["prompt_monitor_reasoning"] = reasoning
                        elif isinstance(prompt_monitor, str) and prompt_monitor.strip():
                            conv_turn["prompt_monitor_reasoning"] = prompt_monitor
                    conversation_history_formatted.append(conv_turn)
            
            # Build user prompt with full state
            state_summary = {
                "user_query": user_query,
                "conversation_history": conversation_history_formatted,  # Previous conversation turns
                "planning": {
                    "plan_quality": state.get("plan_quality", ""),
                    "plan_explanation": state.get("plan_explain", ""),
                    "sql_generated": plan_sql,
                    "clarification_questions": state.get("clarification_questions", [])
                },
                "execution": {
                    "status": "success" if not execution_stats.get("error") else f"failed: {execution_stats.get('error')}",
                    "rows_returned": exec_result.get("row_count", 0),
                    "execution_time_ms": execution_stats.get("execution_time_ms", 0),
                    "columns": columns
                },
                "evaluation": {
                    "satisfaction": state.get("satisfaction", ""),
                    "evaluator_notes": state.get("evaluator_notes", ""),
                    "issues_detected": state.get("issues_detected", [])
                },
                "control_flow": {
                    "clarify_turns": metrics.get("clarify_turns", 0),
                    "total_duration_ms": total_ms,
                    "last_node": state.get("last_node", "unknown")
                },
                "docs_meta": state.get("docs_meta", []),  # Data dictionary for context
                "table_schema": state.get("table_schema", {}),
                "procedural_knowledge": procedural_knowledge
            }
            
            # Get user prompt template from config
            user_template = cfg.get_nested("prompts", "end_prompt_monitor_user_template", default="""Document the complete reasoning process for this agent execution.

Full State Summary:
{state_summary}

Generate a procedural summary explaining what was done and why at each step. Use the data dictionary (docs_meta) to explain table/column choices.""")
            
            user_prompt = user_template.format(
                state_summary=json.dumps(state_summary, indent=2)
            )
            
            logger.info(f"[END_NODE] Calling LLM for prompt monitor with prompt length: {len(user_prompt)} chars")
            try:
                # Use streaming if callback provided, otherwise use regular invoke
                if stream_callback_prompt_monitor:
                    full_response = ""
                    for chunk in llm_client.stream_with_prompt(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.1,  # Lower temperature for more factual reasoning
                        callback=stream_callback_prompt_monitor
                    ):
                        full_response += chunk
                    llm_prompt_monitor = full_response
                else:
                    llm_prompt_monitor = llm_client.invoke_with_prompt(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.1  # Lower temperature for more factual reasoning
                    )
                logger.info(f"[END_NODE] LLM prompt monitor received: {len(llm_prompt_monitor) if llm_prompt_monitor else 0} chars")
                logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM prompt monitor generated successfully ({len(llm_prompt_monitor) if llm_prompt_monitor else 0} chars)"})
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"[END_NODE] LLM prompt monitor generation failed: {str(e)}\n{error_trace}")
                logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM prompt monitor generation failed: {str(e)}", "level": "error", "traceback": error_trace})
                llm_prompt_monitor = None
        else:
            logger.warning("[END_NODE] LLM not available for prompt monitor, using template")
            logs.append({"node": "end", "timestamp": _now_iso(), "msg": "LLM not available, using template prompt monitor"})
    except Exception as e:
        import traceback
        import logging
        logger = logging.getLogger(__name__)
        error_trace = traceback.format_exc()
        logger.error(f"[END_NODE] LLM client error for prompt monitor: {str(e)}\n{error_trace}")
        logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"LLM client error: {str(e)}", "level": "error", "traceback": error_trace})
        llm_prompt_monitor = None
    
    # Fallback to structured template if LLM failed
    if not llm_prompt_monitor:
        # Re-extract conversation history for fallback (list format)
        conversation_history_raw_fallback = state.get("conversation_history_raw", [])
        conversation_history_list = []
        if conversation_history_raw_fallback:
            for i, turn in enumerate(conversation_history_raw_fallback, 1):
                conv_entry = {
                    "turn_number": i,
                    "user_query": turn.get("query", ""),
                    "sql_executed": turn.get("sql", "") if turn.get("sql") and turn.get("sql") != "-- KNOWLEDGE QUESTION" else None,
                    "response_summary": turn.get("response", "")[:200] + "..." if len(turn.get("response", "")) > 200 else turn.get("response", ""),
                    "row_count": turn.get("row_count", 0)
                }
                # Include prompt monitor if available
                prompt_monitor = turn.get('prompt_monitor')
                if prompt_monitor:
                    if isinstance(prompt_monitor, dict):
                        reasoning = prompt_monitor.get('procedural_reasoning', '') or prompt_monitor.get('reasoning', '')
                        if reasoning:
                            conv_entry["prompt_monitor_reasoning"] = reasoning
                    elif isinstance(prompt_monitor, str) and prompt_monitor.strip():
                        conv_entry["prompt_monitor_reasoning"] = prompt_monitor
                conversation_history_list.append(conv_entry)
        
        prompt_monitor = {
            "user_input": user_query,
            "conversation_history": conversation_history_list,  # Previous conversation turns
            "plan_explanation": state.get("plan_explain", ""),
            "execution_summary": {
                "query": plan_sql,
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
    else:
        # LLM-generated prompt monitor (structured)
        prompt_monitor = {
            "procedural_reasoning": llm_prompt_monitor,  # LLM-generated summary
            "raw_state": {  # Keep raw state for reference
                "user_input": user_query,
                "conversation_history": conversation_history_formatted,  # Previous conversation turns
                "plan_explanation": state.get("plan_explain", ""),
                "execution_summary": {
                    "query": plan_sql,
                    "status": "success" if exec_result and not execution_stats.get("error") else "failed",
                    "error": execution_stats.get("error"),
                    "row_count": exec_result.get("row_count", 0),
                    "column_count": len(columns),
                    "duration_ms": execution_stats.get("execution_time_ms", 0)
                },
                "evaluation_notes": state.get("evaluator_notes", ""),
                "satisfaction": state.get("satisfaction", ""),
                "total_agent_duration_ms": total_ms,
                "clarify_turns": metrics.get("clarify_turns", 0)
            },
            "full_logs": logs
        }
    
    logs.append({"node": "end", "timestamp": _now_iso(), "msg": f"agent completed in {total_ms}ms"})
    
    # ============================================================================
    # Final Output Structure
    # ============================================================================
    # CRITICAL: Verify raw_table still has correct data before putting in final_output
    logger.info(f"[END_NODE] Before final_output: raw_table row_count={raw_table.get('row_count', 'MISSING')}, columns={len(raw_table.get('columns', []))}, rows={len(raw_table.get('rows', []))}, query={raw_table.get('query', 'MISSING')[:100] if raw_table.get('query') else 'MISSING'}")
    
    final_output = {
        "raw_table": raw_table,           # OUTPUT 1: Raw table data
        "response": llm_response,         # OUTPUT 2: LLM-generated insights
        "prompt_monitor": prompt_monitor   # OUTPUT 3: LLM-generated reasoning
    }
    
    # Debug: Verify final_output structure
    logger.info(f"[END_NODE] Final output raw_table: row_count={final_output['raw_table'].get('row_count', 'MISSING')}, columns={len(final_output['raw_table'].get('columns', []))}, rows={len(final_output['raw_table'].get('rows', []))}, has_query={bool(final_output['raw_table'].get('query'))}")
    
    # Update conversation history (short-term memory)
    # Check if query was cancelled - if so, don't save to conversation history
    query_cancelled = state.get("query_cancelled", False)
    
    # Use conversation_history_raw (list) if available, otherwise initialize as empty list
    conversation_history = state.get("conversation_history_raw", [])
    if not isinstance(conversation_history, list):
        conversation_history = []
    
    # Only save to conversation history if query was not cancelled
    if not query_cancelled:
        # CRITICAL: Store the ORIGINAL planned SQL (plan_sql) for conversation history
        # This is the SQL the LLM generated, which follow-up queries need to filter.
        # The executed SQL might be sanitized/modified by the safety tool.
        logger.info(f"[END_NODE] Storing SQL in conversation history: plan_sql={plan_sql[:100] if plan_sql else 'N/A'}...")
        logger.info(f"[END_NODE] Executed SQL (may differ): {executed_sql[:100] if executed_sql else 'N/A'}...")
        conversation_entry = {
            "timestamp": _now_iso(),
            "query": user_query,
            "sql": actual_sql,  # Use original planned SQL (plan_sql) for follow-up filtering
            "plan_quality": state.get("plan_quality", ""),
            "response": llm_response,
            "satisfaction": state.get("satisfaction", ""),
            "row_count": exec_result.get("row_count", 0),
            "execution_error": execution_stats.get("error"),
            "prompt_monitor": prompt_monitor,
            "raw_table": raw_table
        }
        conversation_history.append(conversation_entry)
        
        # Keep only last 5 interactions to prevent memory bloat
        conversation_history = conversation_history[-5:]
    else:
        logger.info(f"[END_NODE] Query was cancelled, skipping conversation history save")
    
    # Human-friendly reasoning for end node
    if knowledge_response:
        friendly_reasoning = "Answered from documentation"
    elif exec_result and exec_result.get("row_count", 0) > 0:
        friendly_reasoning = "Preparing your answer"
    else:
        friendly_reasoning = "Finalizing response"
    
    return {
        "last_node": "end",
        "node_reasoning": friendly_reasoning,
        "raw_table": raw_table,  # Add to state
        "final_output": final_output,
        "conversation_history": conversation_history,  # This is actually the raw list (confusing naming, but kept for backward compatibility)
        "conversation_history_raw": conversation_history,  # CRITICAL: Also return as conversation_history_raw so next query can access it
        "control": "end",
        "metrics": metrics,
        "logs": logs,
    }


# Alias for new workflow
def synthesize_response_node(state: AgentState, cfg: QueryAgentConfig, stream_callback_response: Optional[Callable[[str], None]] = None, stream_callback_prompt_monitor: Optional[Callable[[str], None]] = None) -> PartialState:
    """
    Synthesize final response with insights (alias to end_node for new workflow).
    """
    # Add reasoning for synthesize step before calling end_node
    result = end_node(state, cfg, stream_callback_response, stream_callback_prompt_monitor)
    
    # Override reasoning for synthesize step (more specific than end)
    exec_result = state.get("execution_result") or {}
    knowledge_response = state.get("execution_stats", {}).get("knowledge_response")
    
    if knowledge_response:
        synthesize_reasoning = "Preparing answer from documentation"
    elif exec_result and exec_result.get("row_count", 0) > 0:
        synthesize_reasoning = "Analyzing results and preparing your answer"
    else:
        synthesize_reasoning = "Preparing response"
    
    # Update the result to include synthesize reasoning
    if isinstance(result, dict):
        result["node_reasoning"] = synthesize_reasoning
        result["last_node"] = "synthesize"  # Mark as synthesize for UI display
    
    return result


