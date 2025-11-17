from typing import Any, Dict, List, Literal, Optional, TypedDict


ControlSignal = Literal[
    "discover_sources",
    "extract_schemas",
    "extract_text_values",
    "extract_entities",
    "capture_intent",
    "create_embeddings",
    "map_entities",
    "search_entities",
    "decompose_query",
    "construct_queries",
    "validate_queries",
    "tool_decision",
    "select_tool",
    "generate_reasoning",
    "synthesize",
    "end",
]


class AgentState(TypedDict, total=False):
    """
    Typed representation of the Deep Research Agent state passed between nodes.
    """

    # ------------------------------------------------------------------
    # Session context
    # ------------------------------------------------------------------
    user_input: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: Optional[str]

    conversation_history: Optional[List[Dict[str, Any]]]
    conversation_history_raw: Optional[List[Dict[str, Any]]]

    # ------------------------------------------------------------------
    # Control flow metadata
    # ------------------------------------------------------------------
    control: Optional[ControlSignal]
    last_node: Optional[str]
    node_reasoning: Optional[str]
    tool_loop_count: Optional[int]

    # ------------------------------------------------------------------
    # Discovery & schema data
    # ------------------------------------------------------------------
    data_sources: List[Dict[str, Any]]
    table_schemas: Dict[str, Any]
    text_column_values: Dict[str, Any]
    docs_meta: List[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Embeddings & entity mapping
    # ------------------------------------------------------------------
    embeddings_index_path: Optional[str]
    entity_mappings: Dict[str, Any]
    entity_search_results: Dict[str, Any]
    entity_value_samples: Dict[str, Any]

    # ------------------------------------------------------------------
    # Query understanding
    # ------------------------------------------------------------------
    extracted_entities: List[Dict[str, Any]]
    intent_snapshot: Dict[str, Any]
    query_decomposition: Dict[str, Any]
    sql_queries: List[Dict[str, Any]]
    execution_plan: Dict[str, Any]

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    analytical_reasoning: Dict[str, Any]
    final_output: Dict[str, Any]

    # ------------------------------------------------------------------
    # Telemetry & diagnostics
    # ------------------------------------------------------------------
    logs: List[Dict[str, Any]]
    metrics: Dict[str, Any]


PartialState = Dict[str, Any]
