from typing import TypedDict, List, Optional, Dict, Any, Literal


ControlSignal = Literal["plan", "clarify", "replan", "execute", "evaluate", "wait_for_user", "end"]
PlanQuality = Literal["high", "low", "error"]
Satisfaction = Literal["satisfied", "needs_work", "failed"]


class AgentState(TypedDict, total=False):
    # immutable-ish inputs
    user_input: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: Optional[str]

    # enrichment
    docs_meta: List[Dict[str, Any]]
    table_schema: Dict[str, Any]
    parquet_location: str

    # planning
    plan: Optional[Dict[str, Any]]
    plan_quality: Optional[PlanQuality]
    plan_explain: Optional[str]

    # execution
    execution_result: Optional[Dict[str, Any]]  # columns, rows, row_count, execution_time_ms, query
    execution_stats: Optional[Dict[str, Any]]

    # control
    control: Optional[ControlSignal]
    last_node: Optional[str]

    # clarify UX
    clarify_prompt: Optional[str]
    clarify_questions: Optional[List[str]]  # specific questions to ask
    clarify_reasoning: Optional[List[str]]  # reasons for asking
    user_clarification: Optional[str]

    # evaluation
    satisfaction: Optional[Satisfaction]
    evaluator_notes: Optional[str]

    # short-term memory (within current session only)
    conversation_history: Optional[List[Dict[str, str]]]  # [{"query": "...", "response": "..."}]

    # telemetry
    logs: Optional[List[Dict[str, Any]]]

    # metrics
    metrics: Optional[Dict[str, Any]]  # node_timings, total_ms, plan_id, row_count, column_count, clarify_turns

    # raw table output
    raw_table: Optional[Dict[str, Any]]  # columns, rows, row_count, query

    # final outputs
    final_output: Optional[Dict[str, Any]]  # { raw_table: dict, response: str, prompt_monitor: str }


PartialState = Dict[str, Any]


