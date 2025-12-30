"""
Web Research Executor Graph - LangGraph implementation for Web Research Executor pipeline.
"""

import logging
from typing import Dict, Any, Optional
from external.agent.state_types import ResearchExecutorState, ResearchControllerState

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available, will use fallback implementation")


def build_web_research_executor_graph(tools_registry, domain_md: str):
    """
    Build the Web Research Executor LangGraph.
    
    Args:
        tools_registry: Tools registry instance
        domain_md: Domain markdown content
        
    Returns:
        Compiled graph
    """
    if not LANGGRAPH_AVAILABLE:
        return None
    
    graph = StateGraph(ResearchExecutorState)
    
    # Import nodes (will be created in web_research_executor_nodes.py)
    try:
        from external.agent.web_research_executor_nodes import (
            research_discovery_node,
            claim_extraction_node,
            conflict_detection_node,
            evidence_synthesis_node,
            outcome_node
        )
    except ImportError:
        logger.error("Web research executor nodes not available")
        return None
    
    # Add nodes with tools_registry and domain_md bound
    def discovery(state):
        return research_discovery_node(state, tools_registry)
    
    def claim_extraction(state):
        return claim_extraction_node(state, tools_registry, domain_md)
    
    def conflict_detection(state):
        return conflict_detection_node(state, tools_registry, domain_md)
    
    def evidence_synthesis(state):
        return evidence_synthesis_node(state, tools_registry, domain_md)
    
    def outcome(state):
        return outcome_node(state)
    
    graph.add_node("research_discovery", discovery)
    graph.add_node("claim_extraction", claim_extraction)
    graph.add_node("conflict_detection", conflict_detection)
    graph.add_node("evidence_synthesis", evidence_synthesis)
    graph.add_node("outcome", outcome)
    
    # Linear edges
    graph.set_entry_point("research_discovery")
    graph.add_edge("research_discovery", "claim_extraction")
    graph.add_edge("claim_extraction", "conflict_detection")
    graph.add_edge("conflict_detection", "evidence_synthesis")
    graph.add_edge("evidence_synthesis", "outcome")
    graph.add_edge("outcome", END)
    
    return graph.compile()


def run_web_research_executor_sequential(decider_output: dict, state: ResearchControllerState, tools_registry, domain_md: str) -> dict:
    """
    Fallback sequential executor (when LangGraph not available).
    Runs nodes in order.
    """
    # Initialize executor state
    executor_state: ResearchExecutorState = {
        "research_spec": decider_output.get("research_spec", {}),
        "research_spec_status": decider_output.get("research_spec_status", {}),
        "research_plan": decider_output.get("research_spec", {}).get("plan", []),
        "evidence_pack": {"sources": [], "claims": [], "conflicts": [], "gaps": []},
        "sources_seen": state.get("sources_seen", []),
        "executor_report": {},
        "policy_limits": state.get("policy_limits", {}),
        "halt_execution": False,
        "user_query": state.get("user_query"),
        "conversation_history": state.get("conversation_history", []),
        "tavily_api_key": state.get("tavily_api_key"),
        "agent_id": state.get("agent_id"),
        "last_error": None,
    }
    
    try:
        from external.agent.web_research_executor_nodes import (
            research_discovery_node,
            claim_extraction_node,
            conflict_detection_node,
            evidence_synthesis_node,
            outcome_node
        )
    except ImportError:
        logger.error("Web research executor nodes not available")
        return {
            "status": "ERROR",
            "error_type": "NODES_NOT_AVAILABLE",
            "last_error": "Web research executor nodes not implemented"
        }
    
    # Run nodes sequentially
    updates = research_discovery_node(executor_state, tools_registry)
    executor_state.update(updates)
    
    if not executor_state.get("halt_execution"):
        updates = claim_extraction_node(executor_state, tools_registry, domain_md)
        executor_state.update(updates)
    
    if not executor_state.get("halt_execution"):
        updates = conflict_detection_node(executor_state, tools_registry, domain_md)
        executor_state.update(updates)
    
    if not executor_state.get("halt_execution"):
        updates = evidence_synthesis_node(executor_state, tools_registry, domain_md)
        executor_state.update(updates)
    
    # Always run outcome node
    updates = outcome_node(executor_state)
    executor_state.update(updates)
    
    return executor_state.get("executor_report", {})


def run_web_research_executor(decider_output: dict, state: ResearchControllerState, tools_registry, domain_md: str) -> dict:
    """
    Runs Web Research Executor subgraph exactly once.
    Returns schema-valid executor_report with evidence_pack.
    
    Args:
        decider_output: Decider output with research_spec, plan
        state: Research controller state
        tools_registry: Tools registry instance
        domain_md: Domain markdown content
        
    Returns:
        Executor report dictionary with evidence_pack
    """
    if LANGGRAPH_AVAILABLE:
        graph = build_web_research_executor_graph(tools_registry, domain_md)
        if graph:
            # Initialize executor state
            executor_state: ResearchExecutorState = {
                "research_spec": decider_output.get("research_spec", {}),
                "research_spec_status": decider_output.get("research_spec_status", {}),
                "research_plan": decider_output.get("research_spec", {}).get("plan", []),
                "evidence_pack": {"sources": [], "claims": [], "conflicts": [], "gaps": []},
                "sources_seen": state.get("sources_seen", []),
                "executor_report": {},
                "policy_limits": state.get("policy_limits", {}),
                "halt_execution": False,
                "user_query": state.get("user_query"),
                "conversation_history": state.get("conversation_history", []),
                "tavily_api_key": state.get("tavily_api_key"),
                "agent_id": state.get("agent_id"),
                "last_error": None,
            }
            
            # Run graph
            final_state = graph.invoke(executor_state)
            return final_state.get("executor_report", {})
    
    # Fallback to sequential
    return run_web_research_executor_sequential(decider_output, state, tools_registry, domain_md)

