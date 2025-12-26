"""
Executor Graph - LangGraph implementation for Executor pipeline.
"""

import logging
from typing import Dict, Any, Optional
from external.agent.state_types import ExecutorState, ControllerState
from external.agent.executor_nodes import (
    investigation_node,
    sql_generation_node,
    safety_validation_node,
    execution_node,
    evaluation_node,
    outcome_node
)

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available, will use fallback implementation")


def build_executor_graph(tools_registry, domain_md: str):
    """
    Build the Executor LangGraph with 6 linear nodes.
    
    Args:
        tools_registry: Tools registry instance
        domain_md: Domain markdown content
        
    Returns:
        Compiled graph
    """
    if not LANGGRAPH_AVAILABLE:
        # Fallback: return a simple function that runs nodes sequentially
        return None
    
    graph = StateGraph(ExecutorState)
    
    # Add nodes with tools_registry and domain_md bound
    def investigation(state):
        return investigation_node(state, tools_registry)
    
    def sql_gen(state):
        return sql_generation_node(state, tools_registry, domain_md)
    
    def safety(state):
        return safety_validation_node(state, tools_registry)
    
    def execution(state):
        return execution_node(state, tools_registry)
    
    def evaluation(state):
        return evaluation_node(state, tools_registry)
    
    def outcome(state):
        return outcome_node(state)
    
    graph.add_node("investigation", investigation)
    graph.add_node("sql_generation", sql_gen)
    graph.add_node("safety_validation", safety)
    graph.add_node("execution", execution)
    graph.add_node("evaluation", evaluation)
    graph.add_node("outcome", outcome)
    
    # Linear edges
    graph.set_entry_point("investigation")
    graph.add_edge("investigation", "sql_generation")
    graph.add_edge("sql_generation", "safety_validation")
    graph.add_edge("safety_validation", "execution")
    graph.add_edge("execution", "evaluation")
    graph.add_edge("evaluation", "outcome")
    graph.add_edge("outcome", END)
    
    return graph.compile()


def run_executor_sequential(decider_output: dict, state: ControllerState, tools_registry, domain_md: str) -> dict:
    """
    Fallback sequential executor (when LangGraph not available).
    Runs nodes in order.
    """
    # Initialize executor state
    executor_state: ExecutorState = {
        "query_spec": decider_output.get("query_spec", {}),
        "query_spec_status": decider_output.get("query_spec_status", {}),
        "investigation_plan": decider_output.get("investigation_plan", []),
        "final_sql": None,
        "results": None,
        "evaluation": None,
        "executor_report": {},
        "policy_limits": state.get("policy_limits", {}),
        "halt_execution": False
    }
    
    # Run nodes sequentially
    updates = investigation_node(executor_state, tools_registry)
    executor_state.update(updates)
    
    if not executor_state.get("halt_execution"):
        updates = sql_generation_node(executor_state, tools_registry, domain_md)
        executor_state.update(updates)
    
    if not executor_state.get("halt_execution"):
        updates = safety_validation_node(executor_state, tools_registry)
        executor_state.update(updates)
    
    if not executor_state.get("halt_execution"):
        updates = execution_node(executor_state, tools_registry)
        executor_state.update(updates)
    
    if not executor_state.get("halt_execution"):
        updates = evaluation_node(executor_state, tools_registry)
        executor_state.update(updates)
    
    # Always run outcome node
    updates = outcome_node(executor_state)
    executor_state.update(updates)
    
    return executor_state.get("executor_report", {})


def run_executor(decider_output: dict, state: ControllerState, tools_registry, domain_md: str) -> dict:
    """
    Runs Executor subgraph exactly once.
    Returns schema-valid executor_report.
    
    Args:
        decider_output: Decider output with query_spec, investigation_plan
        state: Controller state
        tools_registry: Tools registry instance
        domain_md: Domain markdown content
        
    Returns:
        Executor report dictionary
    """
    if LANGGRAPH_AVAILABLE:
        graph = build_executor_graph(tools_registry, domain_md)
        if graph:
            # Initialize executor state
            executor_state: ExecutorState = {
                "query_spec": decider_output.get("query_spec", {}),
                "query_spec_status": decider_output.get("query_spec_status", {}),
                "investigation_plan": decider_output.get("investigation_plan", []),
                "final_sql": None,
                "results": None,
                "evaluation": None,
                "executor_report": {},
                "policy_limits": state.get("policy_limits", {}),
                "halt_execution": False
            }
            
            # Run graph
            final_state = graph.invoke(executor_state)
            return final_state.get("executor_report", {})
    
    # Fallback to sequential
    return run_executor_sequential(decider_output, state, tools_registry, domain_md)

