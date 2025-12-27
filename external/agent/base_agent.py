from typing import Dict, Any, Optional, List
from tools.base_mcp_tool import BaseMCPTool
from external.agent.parquet_agent import ParquetQueryAgent
from external.agent.config import QueryAgentConfig
from external.agent.parquet_agent import handle_query, load_domain_md


class LangGraphAgentTool(BaseMCPTool):
    """
    MCP-facing facade for the Parquet Query Agent.
    Returns the full final state including dual outputs: response and prompt_monitor.
    """

    def __init__(self, config: Dict = None):
        default_config = {
            "name": "parquet_agent",
            "description": "LangGraph-style agent for planning/clarifying/executing queries on Parquet via DuckDB",
            "version": "1.0.0",
            "enabled": True,
            "inputSchema": {},
            "outputSchema": {},
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.agent = ParquetQueryAgent()
        self.cfg = QueryAgentConfig()
        # Simple in-memory session store for HTTP tool execution path.
        # Keyed by session_id; stores conversation_history and prior_state used by the Decider for follow-ups.
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language request/query"},
                "session_id": {"type": "string", "description": "Optional session id for continuity"},
                "user_clarification": {"type": "string", "description": "User's clarification response (for resuming sessions)"},
                "user_id": {"type": "string", "description": "User ID for session tracking"},
                "show_thinking": {"type": "boolean", "description": "If true, include Decider thinking trace in response"}
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "final_output": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "prompt_monitor": {"type": "string"},
                    },
                },
                "control": {"type": "string"},
                "plan": {"type": "object"},
                "plan_quality": {"type": "string"},
                "plan_explain": {"type": "string"},
                "execution_result": {"type": "object"},
                "execution_stats": {"type": "object"},
                "satisfaction": {"type": "string"},
                "evaluator_notes": {"type": "string"},
                "previous_sessions": {"type": "array"},
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        if not self.enabled:
            raise RuntimeError("Tool is disabled")
        self.validate_arguments(arguments)
        
        query: str = arguments["query"]
        session_id: Optional[str] = arguments.get("session_id")
        user_id: Optional[str] = arguments.get("user_id")  # may be injected by server later
        user_clarification: Optional[str] = arguments.get("user_clarification")
        show_thinking: bool = bool(arguments.get("show_thinking", False))

        # v1: HTTP-only agent/chat. We keep continuity (follow-ups + user answers) via session_id.
        # If no session_id is provided, run stateless.
        effective_query = user_clarification.strip() if isinstance(user_clarification, str) and user_clarification.strip() else query

        conversation_history: List[Dict[str, Any]] = []
        prior_state: Optional[Dict[str, Any]] = None

        if session_id:
            sess = self._sessions.get(session_id) or {}
            conversation_history = list(sess.get("conversation_history") or [])
            prior_state = sess.get("prior_state")

        # Ensure tools are loaded (ParquetQueryAgent loads them in __init__)
        raw = handle_query(
            user_query=effective_query,
            conversation_history=conversation_history,
            prior_state=prior_state,
            tools_registry=self.agent.tools_registry,
            policy_limits={"max_attempts": 3, "max_rows": 1000, "timeout_seconds": 60, "allow_cross_join": False},
            show_thinking=show_thinking,
        )

        # Update session state for continuity
        if session_id:
            # Append a compact turn record (matches what Decider expects)
            turn_status = raw.get("status", "ERROR")
            turn = {
                "query": effective_query,
                "sql": raw.get("final_sql", ""),
                "response": raw.get("result_summary", "") or raw.get("finished_output", "") or raw.get("question", ""),
                "status": turn_status,
            }
            conversation_history.append(turn)

            # Build minimal ControllerState for next follow-up
            domain_md = load_domain_md(effective_query, conversation_history)
            prior_state = {
                "user_query": effective_query,
                "conversation_history": conversation_history,
                "domain_md": domain_md,
                "policy_limits": {"max_attempts": 3, "max_rows": 1000, "timeout_seconds": 60, "allow_cross_join": False},
                "query_spec": raw.get("query_spec", prior_state.get("query_spec", {}) if isinstance(prior_state, dict) else {}),
                "query_spec_status": raw.get("query_spec_status", prior_state.get("query_spec_status", {}) if isinstance(prior_state, dict) else {}),
                "last_executor_report": None,
                "attempt_count": 0,
            }

            self._sessions[session_id] = {
                "conversation_history": conversation_history,
                "prior_state": prior_state,
            }

        # Convert raw handle_query() result to the legacy tool response shape expected by agent_chat.html
        status = raw.get("status")
        if status == "ASK_USER":
            return {
                "session_id": session_id,
                "user_id": user_id,
                "control": "wait_for_user",
                "clarification": {
                    "questions": [raw.get("question", "")],
                    "reasoning": [raw.get("why_non_defaultable", "")],
                    "prompt": raw.get("what_answer_unblocks", ""),
                },
                "query_spec": raw.get("query_spec", {}),
                "query_spec_status": raw.get("query_spec_status", {}),
            }
        if status == "SUCCESS":
            final_output = {
                "response": raw.get("finished_output", ""),
                "sql": raw.get("final_sql", ""),
                "result_summary": raw.get("result_summary", ""),
                "thinking": raw.get("thinking_trace", "") if show_thinking else "",
            }
            # Include results for UI table display
            results = raw.get("results")
            if results:
                final_output["results"] = {
                    "row_count": results.get("row_count", 0),
                    "columns": results.get("columns", []),
                    "rows": results.get("rows_preview", [])
                }
            return {
                "session_id": session_id,
                "user_id": user_id,
                "control": "end",
                "final_output": final_output,
                "plan": {"sql": raw.get("final_sql", "")},
            }
        if status == "BLOCK":
            return {
                "session_id": session_id,
                "user_id": user_id,
                "control": "end",
                "final_output": {"response": f"❌ Query blocked: {raw.get('reason', 'Unknown reason')}"},
            }
        # ERROR
        return {
            "session_id": session_id,
            "user_id": user_id,
            "control": "end",
            "final_output": {"response": f"❌ Error: {raw.get('reason', 'Unknown error')}"},
            "error": raw,
        }


