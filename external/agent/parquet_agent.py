"""
Parquet Agent v1.0 - Decider/Executor Architecture
Controller orchestrates Decider -> Executor -> retry loop
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from external.agent.state_types import ControllerState
from external.agent.decider import run_decider
from external.agent.executor_graph import run_executor
from tools.tools_registry import ToolsRegistry

logger = logging.getLogger(__name__)


def load_domain_md(user_query: str, conversation_history: list, agent_id: Optional[str] = None) -> str:
    """
    Load domain markdown file from agent config, or fallback to query-based selection.
    """
    # If agent_id provided, load from agent config
    if agent_id:
        try:
            from external.agent.agent_registry import get_agent
            agent = get_agent(agent_id)
            if agent:
                domain_file_name = agent.get("domain_file", "")
                if domain_file_name:
                    domain_path = Path(f"external/config/domains/{domain_file_name}")
                    if domain_path.exists():
                        return domain_path.read_text()
                    else:
                        logger.warning(f"Domain file not found for agent {agent_id}: {domain_path}")
        except Exception as e:
            logger.warning(f"Error loading domain for agent {agent_id}: {e}")
    
    # Fallback: Simple domain selection based on query (legacy behavior)
    domain = "ecomm"  # Default domain
    query_lower = user_query.lower()
    if "market risk" in query_lower or "limits" in query_lower or "mr" in query_lower:
        domain = "mr"
    
    domain_file = Path(f"external/config/domains/{domain}_domain.md")
    if domain_file.exists():
        return domain_file.read_text()
    else:
        logger.warning(f"Domain file not found: {domain_file}, using empty domain")
        return ""


def render_ask_user(decider_output: dict) -> dict:
    """Return response to UI: question + context fields."""
    return {
        "status": "ASK_USER",
        "question": decider_output.get("ask_user", {}).get("question", ""),
        "why_non_defaultable": decider_output.get("ask_user", {}).get("why_non_defaultable", ""),
        "what_answer_unblocks": decider_output.get("ask_user", {}).get("what_answer_unblocks", ""),
        "query_spec": decider_output.get("query_spec", {}),
        "query_spec_status": decider_output.get("query_spec_status", {})
    }


def render_block(decider_output: dict) -> dict:
    """Return blocking response to UI."""
    return {
        "status": "BLOCK",
        "reason": decider_output.get("block_reason", "Blocked."),
        "query_spec": decider_output.get("query_spec", {}),
        "query_spec_status": decider_output.get("query_spec_status", {})
    }


def render_success(executor_report: dict, state: Optional[ControllerState] = None) -> dict:
    """Return final user-facing output."""
    result = {
        "status": "SUCCESS",
        "finished_output": executor_report.get("finished_output", ""),
        "final_sql": executor_report.get("final_sql", ""),
        "result_summary": executor_report.get("result_summary", ""),
        "evaluation": executor_report.get("evaluation", {}),
        "results": executor_report.get("results")  # Include results for UI table display
    }
    # Include query_spec and query_spec_status for building prior_state in next query
    if state:
        result["query_spec"] = state.get("query_spec", {})
        result["query_spec_status"] = state.get("query_spec_status", {})
        # Optional UI trace: decider thinking (only when requested)
        if state.get("show_thinking") and state.get("thinking_trace"):
            result["thinking_trace"] = state.get("thinking_trace", "")
    return result


def render_error_max_attempts(last_report: dict, attempt_count: int, max_attempts: int) -> dict:
    """Return error response when max attempts reached."""
    return {
        "status": "ERROR",
        "reason": "Max attempts reached.",
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "last_executor_report": last_report
    }


def initialize_controller_state(
    user_query: str,
    conversation_history: list,
    prior_state: Optional[ControllerState] = None,
    policy_limits: Optional[dict] = None,
    show_thinking: bool = False,
    agent_id: Optional[str] = None
) -> ControllerState:
    """
    Initialize or restore controller state.
    
    Limits conversation_history to last 5 turns to prevent prompt bloat.
    
    Note on conversation_history vs prior_query_spec:
    - conversation_history: List of previous query/response pairs (up to 5 turns)
      Format: [{"query": "...", "sql": "...", "response": "...", "status": "..."}, ...]
      Used by Decider for context and follow-up detection signals.
    
    - prior_query_spec: Single object from the MOST RECENT query only (not a list)
      Format: {"business_question": "...", "dimensions": [...], "metrics": [...], ...}
      Used by Decider as baseline for merging in FOLLOW_UP queries.
    
    So: conversation_history = multiple turns (history), prior_query_spec = latest one (baseline).
    """
    # Limit conversation_history to last 5 turns to prevent prompt bloat
    MAX_CONVERSATION_HISTORY = 5
    if isinstance(conversation_history, list) and len(conversation_history) > MAX_CONVERSATION_HISTORY:
        conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY:]
        logger.info(f"Limited conversation_history to last {MAX_CONVERSATION_HISTORY} turns")
    
    if prior_state:
        # Restore from prior state but UPDATE user_query to the new query
        # This is critical for follow-up queries
        prior_state["user_query"] = user_query
        prior_state["conversation_history"] = conversation_history
        prior_state["show_thinking"] = bool(show_thinking)
        prior_state["thinking_trace"] = None
        # IMPORTANT: attempt_count / last_executor_report are controller-loop fields for a *single* query.
        # When we start a new user query (even a follow-up), we must reset them; otherwise we can
        # incorrectly hit max_attempts immediately using the prior query's state.
        prior_state["attempt_count"] = 0
        prior_state["last_executor_report"] = None
        return prior_state
    
    # Default policy limits
    default_policy_limits = {
        "max_attempts": 3,
        "max_rows": 5000,
        "timeout_seconds": 30,
        "allow_cross_join": False
    }
    
    if policy_limits:
        default_policy_limits.update(policy_limits)
    
    # Load agent config if agent_id provided
    agent_data_folder = None
    if agent_id:
        try:
            from external.agent.agent_registry import get_agent
            agent = get_agent(agent_id)
            if agent:
                agent_data_folder = agent.get("data_folder")
        except Exception as e:
            logger.warning(f"Error loading agent config for {agent_id}: {e}")
    
    return {
        "user_query": user_query,
        "conversation_history": conversation_history,
        "domain_md": load_domain_md(user_query, conversation_history, agent_id),
        "policy_limits": default_policy_limits,
        "query_spec": {},
        "query_spec_status": {},
        "show_thinking": bool(show_thinking),
        "thinking_trace": None,
        "last_executor_report": None,
        "attempt_count": 0,
        "agent_id": agent_id,  # Store agent_id in state
        "agent_data_folder": agent_data_folder  # Store data folder for path scoping
    }


def handle_query(
    user_query: str,
    conversation_history: list,
    prior_state: Optional[ControllerState] = None,
    tools_registry: Optional[ToolsRegistry] = None,
    policy_limits: Optional[dict] = None,
    show_thinking: bool = False,
    agent_id: Optional[str] = None
) -> dict:
    """
    Controller orchestrates:
      Decider -> Executor -> (ERROR -> Decider retry) until SUCCESS / ASK_USER / BLOCK / max_attempts.
    
    Args:
        user_query: User's query string
        conversation_history: List of previous conversation turns
        prior_state: Optional prior controller state (for resuming)
        tools_registry: Tools registry instance (creates default if None)
        policy_limits: Optional policy limits override
        
    Returns:
        Response dictionary with status and results
    """
    # Initialize tools registry if not provided
    if tools_registry is None:
        tools_registry = ToolsRegistry()
        # Load parquet agent tools
        tools_config_dir = Path("external/config/tools")
        for tool_config in tools_config_dir.glob("*.json"):
            if tool_config.stem in [
                "list_dir", "inspect_table", "preview_rows", "search_glossary",
                "nl_to_sql_planner", "sql_plan_updater", "query_safety_validator",
                "execute_sql", "query_result_evaluator"
            ]:
                try:
                    tools_registry.load_tool_from_config(tool_config)
                except Exception as e:
                    logger.warning(f"Failed to load tool {tool_config.stem}: {e}")
    
    # Initialize controller state
    state = initialize_controller_state(user_query, conversation_history, prior_state, policy_limits, show_thinking=show_thinking, agent_id=agent_id)
    max_attempts = int(state["policy_limits"]["max_attempts"])
    domain_md = state["domain_md"]
    
    logger.info(f"Starting query handling: query='{user_query[:50]}...', max_attempts={max_attempts}")
    
    # Main loop — controller-owned retries only
    while True:
        # Enforce max_attempts BEFORE calling executor (attempt_count counts executor runs)
        if state["attempt_count"] >= max_attempts and state["last_executor_report"] is not None:
            logger.warning(f"Max attempts ({max_attempts}) reached")
            return render_error_max_attempts(
                state["last_executor_report"],
                state["attempt_count"],
                max_attempts
            )
        
        # 2A) Call Decider (one prompt call)
        try:
            decider_output = run_decider(state)  # schema-valid decider_output
        except Exception as e:
            logger.error(f"Decider failed: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "reason": f"Decider failed: {str(e)}",
                "error_type": "DECIDER_ERROR"
            }
        
        action = decider_output.get("action")
        query_type = decider_output.get("query_type", "NEW_QUERY")
        logger.info(f"Decider action: {action}, query_type: {query_type}")
        
        # Get new spec from Decider
        new_query_spec = decider_output.get("query_spec", {})
        new_query_spec_status = decider_output.get("query_spec_status", {})
        
        # For FOLLOW_UP queries, preserve missing fields from prior state
        if query_type == "FOLLOW_UP" and state.get("query_spec"):
            prior_spec = state["query_spec"]
            prior_status = state.get("query_spec_status", {})
            
            # Preserve grain if not set
            if not new_query_spec.get("grain") and prior_spec.get("grain"):
                new_query_spec["grain"] = prior_spec["grain"]
                logger.info(f"Preserved grain from prior: {prior_spec['grain']}")
            
            # Preserve start_table if not set
            if not new_query_spec.get("start_table", {}).get("path") and prior_spec.get("start_table", {}).get("path"):
                new_query_spec["start_table"] = prior_spec["start_table"]
                logger.info(f"Preserved start_table from prior")
            
            # Preserve business_question if empty
            if not new_query_spec.get("business_question") and prior_spec.get("business_question"):
                # For follow-ups, combine prior question with new context
                new_query_spec["business_question"] = f"Follow-up: {state['user_query']} (based on: {prior_spec['business_question']})"
            
            # Preserve verified status fields from prior
            for field in ["business_question", "output_shape", "start_table_grain", "time", "metrics", "dimensions", "filters", "joins", "aggregation_plan"]:
                prior_field_status = prior_status.get(field, {})
                new_field_status = new_query_spec_status.get(field, {})
                # If prior was verified and new is missing/empty, preserve verified status
                if prior_field_status.get("status") == "verified" and new_field_status.get("status") in ["missing", "", None]:
                    new_query_spec_status[field] = prior_field_status.copy()
                    new_query_spec_status[field]["notes"] = "Preserved from prior query"
        
        # Keep canonical spec in controller state (single contract)
        state["query_spec"] = new_query_spec
        state["query_spec_status"] = new_query_spec_status
        
        # 2B) Route actions
        if action == "ASK_USER":
            # Controller returns question to UI; loop stops until user replies with more info
            logger.info("Decider requested ASK_USER")
            return render_ask_user(decider_output)
        
        if action == "BLOCK":
            logger.info("Decider requested BLOCK")
            return render_block(decider_output)
        
        if action != "EXECUTE":
            # Defensive: schema should prevent this, but keep safe
            logger.error(f"Invalid action from Decider: {action}")
            return {
                "status": "ERROR",
                "reason": f"Invalid action from Decider: {action}",
                "error_type": "INVALID_ACTION"
            }
        
        # 2C) Execute once (increments attempt_count)
        state["attempt_count"] += 1
        logger.info(f"Executing attempt {state['attempt_count']}/{max_attempts}")
        
        try:
            executor_report = run_executor(decider_output, state, tools_registry, domain_md)
            state["last_executor_report"] = executor_report
        except Exception as e:
            logger.error(f"Executor failed: {e}", exc_info=True)
            # Create error report
            executor_report = {
                "status": "ERROR",
                "error_type": "EXECUTOR_ERROR",
                "failed_checklist_items": ["executor_execution"],
                "what_changed": "Executor raised exception",
                "minimal_fix_suggestion": str(e),
                "last_sql": state.get("query_spec", {}).get("final_sql", ""),
                "last_error": str(e)
            }
            state["last_executor_report"] = executor_report
        
        # 2D) Interpret executor outcome
        if executor_report.get("status") == "SUCCESS":
            logger.info("Executor returned SUCCESS")
            logger.debug(f"Executor report has results: {bool(executor_report.get('results'))}")
            if executor_report.get("results"):
                logger.debug(f"Results keys: {list(executor_report.get('results', {}).keys())}")
            success_result = render_success(executor_report, state)
            logger.debug(f"render_success returned keys: {list(success_result.keys())}")
            logger.debug(f"render_success has results: {bool(success_result.get('results'))}")
            return success_result
        
        # executor_report["status"] == "ERROR" => controller loops back to Decider
        # Decider will see last_executor_report and produce a minimal revised plan or ASK_USER/BLOCK.
        logger.info(f"Executor returned ERROR: {executor_report.get('error_type')}, retrying...")
        continue


class ParquetQueryAgent:
    """
    Parquet Agent v1.0 - Wrapper class for backward compatibility.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.tools_registry = ToolsRegistry()
        # Load parquet agent tools
        self._load_tools()
    
    def _load_tools(self):
        """Load parquet agent tools into registry."""
        tools_config_dir = Path("external/config/tools")
        tool_names = [
            "list_dir", "inspect_table", "preview_rows", "search_glossary",
            "nl_to_sql_planner", "sql_plan_updater", "query_safety_validator",
            "execute_sql", "query_result_evaluator"
        ]
        
        for tool_name in tool_names:
            tool_config = tools_config_dir / f"{tool_name}.json"
            if tool_config.exists():
                try:
                    self.tools_registry.load_tool_from_config(tool_config)
                except Exception as e:
                    logger.warning(f"Failed to load tool {tool_name}: {e}")
    
    def run_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for running a query.
        Maintains backward compatibility with existing API.
        """
        # Build conversation history (simplified for now)
        conversation_history = []
        
        # Call handle_query
        result = handle_query(
            user_query=query,
            conversation_history=conversation_history,
            tools_registry=self.tools_registry,
            policy_limits=self.config.get("policy_limits")
        )
        
        logger.info(f"LangGraphAgentTool: handle_query returned keys: {list(result.keys())}")
        logger.info(f"LangGraphAgentTool: handle_query has results: {bool(result.get('results'))}")
        if result.get('results'):
            logger.info(f"LangGraphAgentTool: results type: {type(result.get('results'))}, keys: {list(result.get('results', {}).keys())}")
        
        # Convert result to old format for backward compatibility
        status = result.get("status")
        
        if status == "ASK_USER":
            out = {
                "session_id": session_id,
                "user_id": user_id,
                "control": "wait_for_user",
                "clarification": {
                    "questions": [result.get("question", "")],
                    "reasoning": [result.get("why_non_defaultable", "")],
                    "prompt": result.get("what_answer_unblocks", "")
                },
                "query_spec": result.get("query_spec", {}),
                "query_spec_status": result.get("query_spec_status", {})
            }
            return out
        
        elif status == "SUCCESS":
            final_output = {
                "response": result.get("finished_output", ""),
                "sql": result.get("final_sql", ""),
                "result_summary": result.get("result_summary", "")
            }
            # Include thinking trace if present
            if result.get("thinking_trace"):
                final_output["thinking"] = result.get("thinking_trace", "")
            
            # Include results for UI table display
            results = result.get("results")
            logger.info(f"LangGraphAgentTool: result.get('results'): {bool(results)}, type: {type(results)}")
            if results:
                logger.info(f"LangGraphAgentTool: results keys: {list(results.keys())}")
                final_output["results"] = {
                    "row_count": results.get("row_count", 0),
                    "columns": results.get("columns", []),
                    "rows": results.get("rows_preview", [])
                }
                logger.info(f"LangGraphAgentTool: Added results to final_output. Row count: {results.get('row_count', 0)}")
            else:
                logger.warning(f"LangGraphAgentTool: No results found in result. Result keys: {list(result.keys())}")
            
            out = {
                "session_id": session_id,
                "user_id": user_id,
                "control": "end",
                "final_output": final_output,
                "plan": {
                    "sql": result.get("final_sql", "")
                }
            }
            return out
        
        elif status == "BLOCK":
            out = {
                "session_id": session_id,
                "user_id": user_id,
                "control": "end",
                "final_output": {
                    "response": f"❌ Query blocked: {result.get('reason', 'Unknown reason')}"
                }
            }
            return out
        
        else:  # ERROR
            return {
                "session_id": session_id,
                "user_id": user_id,
                "control": "end",
                "final_output": {
                    "response": f"❌ Error: {result.get('reason', 'Unknown error')}"
                },
                "error": result
            }
