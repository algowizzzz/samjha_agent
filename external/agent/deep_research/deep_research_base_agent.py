from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool

from external.agent.deep_research.deep_research_agent import DeepResearchAgent
from external.agent.deep_research.deep_research_config import DeepResearchAgentConfig


class DeepResearchAgentTool(BaseMCPTool):
    """
    MCP-facing facade for the Deep Research Agent.
    Produces SQL templates plus analytical reasoning without executing queries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "name": "deep_research_agent",
            "description": "Deep research agent that discovers data, plans queries, and produces analytical reasoning.",
            "version": "1.0.0",
            "enabled": True,
            "inputSchema": {},
            "outputSchema": {},
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.agent = DeepResearchAgent()
        self.cfg = DeepResearchAgentConfig()

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language research objective",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session id for continuity",
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID for session tracking",
                },
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "final_output": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "analytical_reasoning": {"type": "object"},
                        "sql_queries": {"type": "array"},
                    },
                },
                "entity_mappings": {"type": "object"},
                "query_decomposition": {"type": "object"},
                "control": {"type": "string"},
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        if not self.enabled:
            raise RuntimeError("Tool is disabled")
        self.validate_arguments(arguments)

        query: str = arguments["query"]
        session_id: Optional[str] = arguments.get("session_id")
        user_id: Optional[str] = arguments.get("user_id")

        state = self.agent.run_query(query=query, session_id=session_id, user_id=user_id)
        return state


