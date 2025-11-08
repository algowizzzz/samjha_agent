from typing import Dict, Any, Optional, List
from tools.base_mcp_tool import BaseMCPTool
from agent.parquet_agent import ParquetQueryAgent
from agent.config import QueryAgentConfig


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

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language request/query"},
                "session_id": {"type": "string", "description": "Optional session id for continuity"},
                "user_clarification": {"type": "string", "description": "User's clarification response (for resuming sessions)"},
                "user_id": {"type": "string", "description": "User ID for session tracking"},
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
        
        # If user_clarification is provided and session_id exists, resume the session
        if user_clarification and session_id:
            state = self.agent.resume_with_clarification(
                session_id=session_id,
                user_clarification=user_clarification,
                user_id=user_id
            )
        else:
            # Start new query
            state = self.agent.run_query(query=query, session_id=session_id, user_id=user_id)
        
        return state


