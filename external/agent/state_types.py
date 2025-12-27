"""
State type definitions for Parquet Agent.
Defines ExecutorState and ControllerState TypedDicts.
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

