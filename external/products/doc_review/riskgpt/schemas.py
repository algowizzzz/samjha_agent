"""
Schema definitions for RiskGPT Agent.
"""
from typing import TypedDict, List, Optional, Dict, Any, Literal


ControlSignal = Literal[
    "context_loader",      # Initial setup - load doc, blocks, template
    "intent_classifier",   # Classify user intent
    "block_improver",      # Generate block improvements
    "chat_responder",      # Answer general questions
    "doc_searcher",        # Search and answer about unselected parts
    "end"                  # Finish
]

IntentType = Literal[
    "improve_blocks",      # User wants to improve selected blocks
    "general_question",    # User asking about doc/template/process
    "search_document",     # User asking about unselected parts of doc
    "compliance_check",    # User wants compliance assessment
    "mixed"                # Multiple intents detected
]


class RiskGPTAgentState(TypedDict, total=False):
    """
    State for RiskGPT Agent workflow.
    
    Flow:
    1. Context Loader (MCP) - loads all document context
    2. Intent Classifier (LLM) - determines what user wants
    3. One of: Block Improver | Chat Responder | Doc Searcher (LLM)
    4. End (MCP) - formats final response
    """
    
    # ============================================================================
    # INPUT (set by caller)
    # ============================================================================
    file_id: str
    user_prompt: str
    selected_block_ids: List[str]  # Empty for general chat
    conversation_history: List[Dict[str, str]]  # Last 5-10 messages
    
    # ============================================================================
    # CONTEXT (loaded by context_loader node)
    # ============================================================================
    full_markdown: str
    block_metadata: List[Dict[str, Any]]
    selected_blocks: List[Dict[str, Any]]  # Filtered from block_metadata
    template_name: Optional[str]
    template_content: Optional[str]
    all_suggestions: List[Dict[str, Any]]  # All pending/accepted/rejected
    
    # ============================================================================
    # INTENT CLASSIFICATION (intent_classifier node)
    # ============================================================================
    intent: Optional[IntentType]
    intent_confidence: Optional[float]  # 0.0 - 1.0
    intent_reasoning: Optional[str]
    requires_block_context: Optional[bool]
    requires_doc_search: Optional[bool]
    
    # ============================================================================
    # BLOCK IMPROVEMENT (block_improver node)
    # ============================================================================
    block_analysis: Optional[str]  # LLM's analysis of what to improve
    block_suggestions: Optional[List[Dict[str, Any]]]  # Structured suggestions
    
    # ============================================================================
    # CHAT RESPONSE (chat_responder node)
    # ============================================================================
    chat_response: Optional[str]  # Conversational markdown response
    referenced_sections: Optional[List[str]]  # Doc sections referenced
    
    # ============================================================================
    # DOC SEARCH (doc_searcher node)
    # ============================================================================
    search_query: Optional[str]  # Extracted search intent
    found_blocks: Optional[List[Dict[str, Any]]]  # Relevant blocks found
    search_response: Optional[str]  # Response with search results
    
    # ============================================================================
    # CONTROL FLOW
    # ============================================================================
    control: Optional[ControlSignal]
    last_node: Optional[str]
    node_reasoning: Optional[str]
    
    # ============================================================================
    # OUTPUTS
    # ============================================================================
    final_output: Optional[Dict[str, Any]]  # { analysis: str, suggestions: [...] }
    
    # ============================================================================
    # TELEMETRY
    # ============================================================================
    logs: Optional[List[Dict[str, Any]]]
    metrics: Optional[Dict[str, Any]]


PartialState = Dict[str, Any]

