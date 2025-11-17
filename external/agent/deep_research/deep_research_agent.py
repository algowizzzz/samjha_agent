"""Deep Research Agent orchestration."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from external.agent.deep_research.deep_research_config import DeepResearchAgentConfig
from external.agent.deep_research.deep_research_nodes import (
    capture_intent_node,
    construct_queries_node,
    create_embeddings_node,
    decompose_query_node,
    discover_sources_node,
    end_node,
    extract_entities_node,
    extract_schemas_node,
    extract_text_values_node,
    generate_reasoning_node,
    map_entities_node,
    search_entities_node,
    select_tool_node,
    synthesize_response_node,
    tool_decision_node,
    validate_queries_node,
)
from external.agent.state_manager import AgentStateManager

logger = logging.getLogger(__name__)


class DeepResearchAgent:
    """Executes the Deep Research workflow by chaining node functions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        self.cfg = DeepResearchAgentConfig(config_path=config_path)
        self.state_manager = AgentStateManager()
        self.max_steps = 50
        self.node_map = {
            "discover_sources": discover_sources_node,
            "extract_schemas": extract_schemas_node,
            "extract_text_values": extract_text_values_node,
            "extract_entities": extract_entities_node,
            "capture_intent": capture_intent_node,
            "create_embeddings": create_embeddings_node,
            "map_entities": map_entities_node,
            "search_entities": search_entities_node,
            "decompose_query": decompose_query_node,
            "construct_queries": construct_queries_node,
            "validate_queries": validate_queries_node,
            "tool_decision": tool_decision_node,
            "select_tool": select_tool_node,
            "generate_reasoning": generate_reasoning_node,
            "synthesize": synthesize_response_node,
            "end": end_node,
        }

    @staticmethod
    def _merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        base.update(updates)
        return base

    def run_query(
        self, query: str, session_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        state: Dict[str, Any] = {
            "user_input": query,
            "user_id": user_id,
            "session_id": sid,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "logs": [],
            "metrics": {"start_time": datetime.utcnow().isoformat() + "Z"},
            "control": "discover_sources",
        }

        steps = 0
        while steps < self.max_steps:
            control = state.get("control", "end")
            node_fn = self.node_map.get(control)
            if not node_fn:
                logger.warning("Unknown control signal %s; terminating", control)
                break
            steps += 1
            try:
                logger.debug("[DeepResearchAgent] Step %s executing node %s", steps, control)
                partial_state = node_fn(state, self.cfg)
                state = self._merge(state, partial_state)
            except Exception as exc:  # pragma: no cover - safety net
                logger.exception("Node %s failed: %s", control, exc)
                state.setdefault("logs", []).append(
                    {
                        "node": control,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "msg": f"node failed: {exc}",
                        "level": "error",
                    }
                )
                state["control"] = "end"
                state = self._merge(state, end_node(state, self.cfg))
                break

            if state.get("control") == "end":
                break

        if state.get("control") != "end":
            state = self._merge(state, end_node(state, self.cfg))

        state.setdefault("metrics", {}).update({"total_steps": steps})
        self.state_manager.save_session_state(sid, user_id, state)
        return state

    # Clarification loop is not used in the deep research workflow.
    def resume_with_clarification(
        self, session_id: str, user_clarification: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError("DeepResearchAgent does not support clarifications")
