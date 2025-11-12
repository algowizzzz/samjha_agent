from typing import TypedDict, List, Optional, Dict, Any, Literal


ControlSignal = Literal[
    "invoke",               # Initial setup
    "check_structure",      # Is query structured/unstructured?
    "check_ambiguity",      # Is query clear?
    "clarify",              # Ask clarification questions
    "process_clarification",# Process user's clarification
    "generate_sql",         # Build SQL
    "execute_sql",          # Run query
    "retry_sql",            # Rebuild SQL after error
    "synthesize",           # Generate final response
    "wait_for_user",        # Pause for user input
    "end"                   # Finish
]
PlanQuality = Literal["high", "medium", "low", "error"]
Satisfaction = Literal["satisfied", "needs_work", "failed"]


class AgentState(TypedDict, total=False):
    # ============================================================================
    # FIXED FIELDS (always present, set early)
    # ============================================================================
    
    # immutable-ish inputs
    user_input: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: Optional[str]

    # enrichment (set by invoke_node)
    docs_meta: List[Dict[str, Any]]
    table_schema: Dict[str, Any]
    parquet_location: str

    # short-term memory (within current session only)
    conversation_history: Optional[List[Dict[str, str]]]  # [{"query": "...", "response": "..."}]
    conversation_history_raw: Optional[List[Dict[str, Any]]]  # Raw conversation data with tables

    # ============================================================================
    # NODE OUTPUTS (structured by node, each node updates its section)
    # ============================================================================
    
    # control flow
    control: Optional[ControlSignal]
    last_node: Optional[str]
    node_reasoning: Optional[str]  # Last node's reasoning
    
    # check_structure_node output
    is_structured: Optional[bool]
    
    # check_ambiguity_node output
    is_ambiguous: Optional[bool]
    ambiguity_reasons: Optional[List[str]]
    clarification_questions: Optional[List[str]]
    
    # process_clarification_node output
    user_clarification: Optional[str]
    clarification_count: Optional[int]
    accumulated_clarifications: Optional[List[str]]
    
    # generate_sql_node output
    plan: Optional[Dict[str, Any]]
    plan_quality: Optional[PlanQuality]
    plan_explain: Optional[str]
    sql_attempt_count: Optional[int]
    
    # execute_sql_node output
    execution_result: Optional[Dict[str, Any]]  # columns, rows, row_count, execution_time_ms, query
    execution_stats: Optional[Dict[str, Any]]
    
    # clarify_node output
    clarify_prompt: Optional[str]
    clarify_reasoning: Optional[List[str]]
    
    # evaluate_node output (if used)
    satisfaction: Optional[Satisfaction]
    evaluator_notes: Optional[str]
    
    # Generic node output (for any node-specific data)
    node_output: Optional[Dict[str, Any]]  # Last node's specific output

    # ============================================================================
    # TELEMETRY & OUTPUTS (sections 10-13)
    # ============================================================================
    
    # telemetry
    logs: Optional[List[Dict[str, Any]]]

    # metrics
    metrics: Optional[Dict[str, Any]]  # node_timings, total_ms, plan_id, row_count, column_count, clarify_turns

    # raw table output
    raw_table: Optional[Dict[str, Any]]  # columns, rows, row_count, query

    # final outputs
    final_output: Optional[Dict[str, Any]]  # { raw_table: dict, response: str, prompt_monitor: str }


PartialState = Dict[str, Any]


