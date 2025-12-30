"""
State type definitions for Parquet Agent and Web Research Agent.
Defines ExecutorState, ControllerState, ResearchExecutorState, and ResearchControllerState TypedDicts.
"""

from typing import TypedDict, Optional, Any


class ExecutorState(TypedDict):
    """State for Executor graph execution."""
    query_spec: dict
    query_spec_status: dict
    investigation_plan: list
    final_sql: Optional[str]
    results: Optional[Any]
    evaluation: Optional[dict]  # Evaluation result from query_result_evaluator
    executor_report: dict
    policy_limits: dict
    halt_execution: bool  # Early halt flag
    user_query: Optional[str]  # User's original query for LLM commentary
    conversation_history: Optional[list]  # Conversation history for context
    agent_data_folder: Optional[str]  # Agent's data folder for path scoping in tools
    last_error: Optional[str]  # Propagated execution/gate error (must be part of state for LangGraph)


class ControllerState(TypedDict):
    """State for Controller loop orchestration."""
    user_query: str
    conversation_history: list
    domain_md: str
    policy_limits: dict
    query_spec: dict
    query_spec_status: dict
    # Optional UI-facing trace (request-scoped; only used when explicitly enabled)
    show_thinking: bool
    thinking_trace: Optional[str]
    last_executor_report: Optional[dict]
    attempt_count: int
    agent_id: Optional[str]  # Agent instance ID
    agent_data_folder: Optional[str]  # Agent's data folder for path scoping


class ResearchExecutorState(TypedDict):
    """State for Web Research Executor graph execution."""
    research_spec: dict
    research_spec_status: dict
    research_plan: list  # Plan from Decider (list of Tavily tool calls)
    evidence_pack: dict  # Accumulated evidence: sources, claims, conflicts, gaps
    sources_seen: list  # URLs already fetched (prevent duplicates)
    executor_report: dict
    policy_limits: dict
    halt_execution: bool  # Early halt flag
    user_query: Optional[str]  # User's original query for LLM commentary
    conversation_history: Optional[list]  # Conversation history for context
    tavily_api_key: Optional[str]  # Tavily API key from agent config
    agent_id: Optional[str]  # Agent instance ID (for loading prompts from DB)
    last_error: Optional[str]  # Propagated execution/gate error (must be part of state for LangGraph)


class ResearchControllerState(TypedDict):
    """State for Web Research Controller loop orchestration."""
    user_query: str
    conversation_history: list
    domain_md: str  # Domain configuration (authority domains, search scope, research depth)
    policy_limits: dict
    research_spec: dict
    research_spec_status: dict
    evidence_pack: Optional[dict]  # Accumulated evidence from prior iterations
    sources_seen: list  # URLs already seen (prevent duplicates across iterations)
    iteration_count: int  # Current iteration number (0-based)
    # Optional UI-facing trace (request-scoped; only used when explicitly enabled)
    show_thinking: bool
    thinking_trace: Optional[str]
    last_executor_report: Optional[dict]
    attempt_count: int
    agent_id: Optional[str]  # Agent instance ID
    tavily_api_key: Optional[str]  # Tavily API key from agent config
    continuity_packet: Optional[dict]  # Continuity packet for cross-turn state

