"""
Nodes for RiskGPT Agent.
Each node is a pure function that takes state and returns partial state updates.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from external.products.doc_review.riskgpt.schemas import RiskGPTAgentState, PartialState
from external.platform.llm import get_llm_client, is_llm_available


class LLMNotAvailableError(RuntimeError):
    """LLM not available error."""
    pass

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _make_node_result(
    state: RiskGPTAgentState,
    node_name: str,
    control: str,
    reasoning: str,
    state_updates: Optional[Dict[str, Any]] = None
) -> PartialState:
    """Standardized node return format."""
    logs = state.get("logs", [])
    logs.append({
        "node": node_name,
        "timestamp": _now_iso(),
        "msg": reasoning,
        "control": control
    })
    
    result = {
        "last_node": node_name,
        "control": control,
        "node_reasoning": reasoning,
        "logs": logs,
    }
    
    if state_updates:
        result.update(state_updates)
    
    return result


# ============================================================================
# NODE 1: Context Loader (MCP - pure data loading)
# ============================================================================

def context_loader_node(
    state: RiskGPTAgentState,
    document_state: Dict[str, Any],
    template_content: Optional[str] = None
) -> PartialState:
    """
    Load all document context needed for RiskGPT.
    Pure data loading - no LLM.
    
    Args:
        state: Current agent state with file_id, selected_block_ids
        document_state: Document state from store
        template_content: Template markdown (optional)
    
    Returns:
        Partial state with loaded context
    """
    full_markdown = document_state.get("raw_markdown", "")
    block_metadata = document_state.get("block_metadata", [])
    template_name = document_state.get("template_name")
    
    # Filter selected blocks
    selected_block_ids = state.get("selected_block_ids", [])
    selected_blocks = []
    if selected_block_ids and block_metadata:
        selected_blocks = [b for b in block_metadata if b.get("id") in selected_block_ids]
    
    # Build suggestions list
    template_improvements = document_state.get("template_improvements", [])
    accepted_suggestions = set(document_state.get("accepted_suggestions", []))
    rejected_suggestions = set(document_state.get("rejected_suggestions", []))
    
    all_suggestions = []
    for imp in template_improvements:
        block_id = imp.get("block_id")
        status = "pending"
        if block_id in accepted_suggestions:
            status = "accepted"
        elif block_id in rejected_suggestions:
            status = "rejected"
        all_suggestions.append({
            "block_id": block_id,
            "status": status,
            "reasoning": imp.get("reasoning", ""),
            "changes_made": imp.get("changes_made", [])
        })
    
    return _make_node_result(
        state,
        "context_loader",
        "intent_classifier",
        f"Loaded context: {len(block_metadata)} blocks, {len(selected_blocks)} selected, {len(all_suggestions)} suggestions",
        {
            "full_markdown": full_markdown,
            "block_metadata": block_metadata,
            "selected_blocks": selected_blocks,
            "template_name": template_name,
            "template_content": template_content,
            "all_suggestions": all_suggestions,
        }
    )


# ============================================================================
# NODE 2: Intent Classifier (LLM - understand what user wants)
# ============================================================================

def intent_classifier_node(state: RiskGPTAgentState) -> PartialState:
    """
    Classify user intent to route to appropriate handler.
    
    Intents:
    - improve_blocks: User wants specific block improvements
    - general_question: User asking about doc/template/process
    - search_document: User asking about unselected parts of doc
    - compliance_check: User wants compliance assessment
    """
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")
    
    client = get_llm_client()
    user_prompt = state.get("user_prompt", "")
    selected_blocks = state.get("selected_blocks", [])
    conversation_history = state.get("conversation_history", [])
    
    # Build context
    has_selected_blocks = len(selected_blocks) > 0
    recent_chat = ""
    if conversation_history:
        recent_chat = "\n".join([
            f"{'Q' if msg.get('role') == 'user' else 'A'}: {msg.get('content', '')}"
            for msg in conversation_history[-3:]
        ])
    
    system_prompt = """You are an intent classifier for RiskGPT, a document review assistant.

Analyze the user's request and classify it into ONE primary intent:

1. **improve_blocks** - User wants to improve/modify selected blocks
   Examples: "make this clearer", "add more detail", "fix grammar", "improve this section"
   
2. **general_question** - User asking about the document, template, or process
   Examples: "what is this document about?", "explain section 3", "what's missing?", "is this compliant?"
   
3. **search_document** - User asking about parts NOT currently selected
   Examples: "find risk disclosures", "where does it mention X?", "show me sections about Y"
   
4. **compliance_check** - User wants compliance/gap analysis
   Examples: "check compliance", "what gaps exist?", "assess against template"

Respond ONLY with valid JSON:
{
  "intent": "improve_blocks|general_question|search_document|compliance_check",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "requires_block_context": true/false,
  "requires_doc_search": true/false
}"""
    
    user_content = f"""USER REQUEST: {user_prompt}

CONTEXT:
- Blocks selected: {"Yes (" + str(len(selected_blocks)) + " blocks)" if has_selected_blocks else "No"}
- Recent conversation: {recent_chat or "None"}

Classify the intent."""
    
    try:
        response = client.invoke(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            temperature=0.1,
            max_tokens=300,
            response_format="json"
        )
        
        # Parse JSON response  
        result = json.loads(response)
        
        intent = result.get("intent", "general_question")
        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")
        requires_block_context = result.get("requires_block_context", has_selected_blocks)
        requires_doc_search = result.get("requires_doc_search", False)
        
        # Route to appropriate node
        if intent == "improve_blocks" and has_selected_blocks:
            next_control = "block_improver"
        elif intent == "search_document" or requires_doc_search:
            next_control = "doc_searcher"
        else:
            next_control = "chat_responder"
        
        logger.info(f"[IntentClassifier] Classified as '{intent}' (confidence: {confidence:.2f}) -> {next_control}")
        
        return _make_node_result(
            state,
            "intent_classifier",
            next_control,
            f"Intent: {intent} (confidence: {confidence:.2f})",
            {
                "intent": intent,
                "intent_confidence": confidence,
                "intent_reasoning": reasoning,
                "requires_block_context": requires_block_context,
                "requires_doc_search": requires_doc_search,
            }
        )
        
    except Exception as e:
        logger.error(f"[IntentClassifier] Error: {e}, falling back to default routing")
        # Fallback: if blocks selected -> improve, else -> chat
        next_control = "block_improver" if has_selected_blocks else "chat_responder"
        return _make_node_result(
            state,
            "intent_classifier",
            next_control,
            f"Fallback routing (error: {str(e)})",
            {
                "intent": "improve_blocks" if has_selected_blocks else "general_question",
                "intent_confidence": 0.5,
                "intent_reasoning": "Fallback due to classification error",
                "requires_block_context": has_selected_blocks,
                "requires_doc_search": False,
            }
        )


# ============================================================================
# NODE 3: Block Improver (LLM - generate block improvements)
# ============================================================================

def block_improver_node(state: RiskGPTAgentState) -> PartialState:
    """
    Generate structured suggestions for selected blocks.
    """
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")
    
    client = get_llm_client()
    selected_blocks = state.get("selected_blocks", [])
    user_prompt = state.get("user_prompt", "")
    template_content = state.get("template_content")
    all_suggestions = state.get("all_suggestions", [])
    
    system_prompt = """You are RiskGPT Block Improver - a specialized assistant for improving policy document blocks.

YOUR TASK:
Generate SPECIFIC, ACTIONABLE improvements for the selected blocks based on user's request.

OUTPUT FORMAT (valid JSON only):
{
  "analysis": "Brief explanation of what needs improvement and why",
  "suggestions": [
    {
      "block_id": "block_id_here",
      "original": "original text",
      "suggested": "improved text",
      "reason": "specific reason for this change",
      "confidence": "high|medium|low"
    }
  ]
}

GUIDELINES:
1. Be specific - reference exact text and changes
2. Provide clear reasoning for each suggestion
3. Consider template requirements if available
4. Don't contradict previously accepted suggestions
5. Focus on the user's specific request"""
    
    # Build context
    blocks_json = json.dumps([
        {
            "block_id": b.get("id"),
            "content": b.get("content"),
            "type": b.get("type", "paragraph")
        }
        for b in selected_blocks
    ], indent=2)
    
    template_context = ""
    if template_content:
        template_context = f"\n\nTEMPLATE REQUIREMENTS:\n{template_content[:2000]}"
    
    suggestions_context = ""
    if all_suggestions:
        pending = [s for s in all_suggestions if s.get('status') == 'pending']
        suggestions_context = f"\n\nEXISTING SUGGESTIONS: {len(all_suggestions)} total ({len(pending)} pending)"
    
    user_content = f"""USER REQUEST: {user_prompt}

SELECTED BLOCKS:
{blocks_json}{template_context}{suggestions_context}

Generate improvements."""
    
    try:
        response = client.invoke(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            temperature=0.3,
            max_tokens=2000,
            response_format="json"
        )
        
        result = json.loads(response)
        
        analysis = result.get("analysis", "")
        suggestions = result.get("suggestions", [])
        
        logger.info(f"[BlockImprover] Generated {len(suggestions)} suggestions")
        
        return _make_node_result(
            state,
            "block_improver",
            "end",
            f"Generated {len(suggestions)} block improvements",
            {
                "block_analysis": analysis,
                "block_suggestions": suggestions,
            }
        )
        
    except Exception as e:
        logger.error(f"[BlockImprover] Error: {e}")
        return _make_node_result(
            state,
            "block_improver",
            "end",
            f"Error generating improvements: {str(e)}",
            {
                "block_analysis": f"Error: {str(e)}",
                "block_suggestions": [],
            }
        )


# ============================================================================
# NODE 4: Chat Responder (LLM - conversational responses)
# ============================================================================

def chat_responder_node(state: RiskGPTAgentState) -> PartialState:
    """
    Provide conversational responses about the document, template, or process.
    """
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")
    
    client = get_llm_client()
    user_prompt = state.get("user_prompt", "")
    full_markdown = state.get("full_markdown", "")
    template_content = state.get("template_content")
    all_suggestions = state.get("all_suggestions", [])
    conversation_history = state.get("conversation_history", [])
    
    system_prompt = """You are RiskGPT - a helpful document review assistant.

YOUR ROLE:
- Answer questions about the document, template, and review process
- Provide guidance and explanations
- Reference specific sections when helpful
- Stay focused on document review

RESPONSE STYLE:
- Conversational and helpful
- Use markdown formatting (headings, lists, bold, code blocks)
- Cite specific sections/pages when relevant
- Be concise but complete

RULES:
1. ONLY help with document review - politely decline unrelated requests
2. Use conversation history for context (marked with Q: and A: prefixes)
3. Be specific and cite document sections
4. Keep responses under 500 words unless detailed analysis is needed
5. NEVER include conversation formatting (Q:, A:, USER:, ASSISTANT:) in your response"""
    
    # Build context
    history_text = ""
    if conversation_history:
        history_text = "[PREVIOUS CONVERSATION CONTEXT]\n" + "\n".join([
            f"{'Q' if msg.get('role') == 'user' else 'A'}: {msg.get('content', '')}"
            for msg in conversation_history[-5:]
        ]) + "\n[END CONTEXT]\n\n"
    
    template_text = ""
    if template_content:
        template_text = f"\n\nTEMPLATE:\n{template_content[:3000]}"
    
    suggestions_text = ""
    if all_suggestions:
        pending = [s for s in all_suggestions if s.get('status') == 'pending']
        accepted = [s for s in all_suggestions if s.get('status') == 'accepted']
        suggestions_text = f"\n\nSUGGESTIONS SUMMARY:\n- Total: {len(all_suggestions)}\n- Pending: {len(pending)}\n- Accepted: {len(accepted)}"
    
    user_content = f"""{history_text}USER QUESTION: {user_prompt}

DOCUMENT CONTENT:
{full_markdown[:15000]}{template_text}{suggestions_text}

Answer the user's question."""
    
    try:
        response = client.invoke(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            temperature=0.5,
            max_tokens=1500
        )
        
        logger.info(f"[ChatResponder] Generated response ({len(response)} chars)")
        
        return _make_node_result(
            state,
            "chat_responder",
            "end",
            "Generated conversational response",
            {
                "chat_response": response,
                "referenced_sections": [],  # Could parse this from response
            }
        )
        
    except Exception as e:
        logger.error(f"[ChatResponder] Error: {e}")
        return _make_node_result(
            state,
            "chat_responder",
            "end",
            f"Error generating response: {str(e)}",
            {
                "chat_response": f"Sorry, I encountered an error: {str(e)}",
                "referenced_sections": [],
            }
        )


# ============================================================================
# NODE 5: Doc Searcher (LLM+MCP - search and answer about unselected parts)
# ============================================================================

def doc_searcher_node(state: RiskGPTAgentState) -> PartialState:
    """
    Search document for relevant content and provide targeted answers.
    """
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")
    
    client = get_llm_client()
    user_prompt = state.get("user_prompt", "")
    full_markdown = state.get("full_markdown", "")
    block_metadata = state.get("block_metadata", [])
    
    system_prompt = """You are RiskGPT Document Searcher.

YOUR TASK:
1. Identify what the user is looking for in the document
2. Search through the document content
3. Provide relevant excerpts and answer their question

OUTPUT: Natural markdown response that:
- Directly answers the user's question
- Quotes relevant sections with page/block references
- Explains what was found (or not found)
- Uses clear formatting"""
    
    user_content = f"""USER REQUEST: {user_prompt}

DOCUMENT CONTENT:
{full_markdown[:20000]}

Find and explain relevant sections."""
    
    try:
        response = client.invoke(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        logger.info(f"[DocSearcher] Generated search response ({len(response)} chars)")
        
        return _make_node_result(
            state,
            "doc_searcher",
            "end",
            "Searched document and generated response",
            {
                "search_response": response,
                "found_blocks": [],  # Could extract mentioned blocks
            }
        )
        
    except Exception as e:
        logger.error(f"[DocSearcher] Error: {e}")
        return _make_node_result(
            state,
            "doc_searcher",
            "end",
            f"Error searching document: {str(e)}",
            {
                "search_response": f"Sorry, I encountered an error while searching: {str(e)}",
                "found_blocks": [],
            }
        )


# ============================================================================
# NODE 6: End (MCP - format final output)
# ============================================================================

def end_node(state: RiskGPTAgentState) -> PartialState:
    """
    Format final output based on which processing path was taken.
    """
    last_node = state.get("last_node", "")
    
    # Determine output based on which node ran
    if "block_improver" in last_node:
        analysis = state.get("block_analysis", "")
        suggestions = state.get("block_suggestions", [])
        final_output = {
            "analysis": analysis,
            "suggestions": suggestions
        }
    elif "chat_responder" in last_node:
        chat_response = state.get("chat_response", "")
        final_output = {
            "analysis": chat_response,
            "suggestions": []
        }
    elif "doc_searcher" in last_node:
        search_response = state.get("search_response", "")
        final_output = {
            "analysis": search_response,
            "suggestions": []
        }
    else:
        final_output = {
            "analysis": "No response generated",
            "suggestions": []
        }
    
    return _make_node_result(
        state,
        "end",
        "end",
        "RiskGPT workflow completed",
        {
            "final_output": final_output
        }
    )

