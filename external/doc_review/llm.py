from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from external.agent.llm_client import get_llm_client, is_llm_available

logger = logging.getLogger(__name__)


class LLMNotAvailableError(RuntimeError):
    pass


def _escape_json_newlines(text: str) -> str:
    result: List[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if ch in ("\r", "\n") and in_string:
            result.append("\\n")
            continue
        result.append(ch)
    return "".join(result)


def call_llm_json(system_prompt: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not is_llm_available():
        logger.warning("LLM not available; returning empty result for prompt: %s", system_prompt[:80])
        raise LLMNotAvailableError("LLM not configured")

    client = get_llm_client()
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    response = client.invoke_with_prompt(system_prompt, user_prompt, response_format="json")

    try:
        # Try to extract JSON from markdown code blocks if present
        cleaned_response = response.strip()
        if cleaned_response.startswith("```"):
            # Extract JSON from code block
            lines = cleaned_response.split("\n")
            if len(lines) > 2:
                # Skip first line (```json or ```) and last line (```)
                cleaned_response = "\n".join(lines[1:-1])
        elif cleaned_response.startswith("```json"):
            lines = cleaned_response.split("\n")
            if len(lines) > 2:
                cleaned_response = "\n".join(lines[1:-1])
        
        try:
            parsed = json.loads(cleaned_response)
        except json.JSONDecodeError:
            sanitized = _escape_json_newlines(cleaned_response)
            parsed = json.loads(sanitized)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode LLM JSON response: %s", exc)
        logger.error("Response preview (first 500 chars): %s", response[:500])
        logger.warning("Returning empty list due to JSON decode error")
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError("Unexpected LLM response format")


_DOC_CHAT_SYSTEM_PROMPT = (
    "You are an expert document review assistant. "
    "Use the provided context (outline, selected section, and document excerpt) to answer the user's question. "
    "Reference headings or sections when relevant. "
    "If the answer is unclear from the supplied text, state what additional information is needed."
)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def generate_chat_reply(
    message: str,
    document_state: Dict[str, Any],
    selection: Optional[str] = None,
    max_excerpt_chars: int = 5000,
    max_selection_chars: int = 1500,
) -> str:
    if not message.strip():
        raise ValueError("message must be provided")

    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")

    client = get_llm_client()

    file_metadata = document_state.get("file_metadata", {}) or {}
    doc_title = file_metadata.get("file_id") or document_state.get("file_id") or "Document"

    toc_entries = document_state.get("index", {}).get("toc", []) or []
    outline_lines: List[str] = []
    for entry in toc_entries[:12]:
        heading = entry.get("heading") or ""
        if heading:
            outline_lines.append(f"- {heading}")
    outline_text = "\n".join(outline_lines)

    selection_snippet = _truncate(selection, max_selection_chars) if selection else ""
    improved_markdown = document_state.get("improved_markdown") or ""
    if not improved_markdown:
        improved_markdown = document_state.get("raw_markdown", "")
    document_excerpt = _truncate(improved_markdown, max_excerpt_chars) if improved_markdown else ""

    user_sections = [
        f"Document: {doc_title}",
        f"Question:\n{message.strip()}",
    ]
    if outline_text:
        user_sections.append(f"Outline:\n{outline_text}")
    if selection_snippet:
        user_sections.append(f"Selected excerpt:\n{selection_snippet}")
    if document_excerpt:
        user_sections.append(f"Document excerpt:\n{document_excerpt}")

    user_prompt = "\n\n".join(user_sections)

    response = client.invoke(
        messages=[{"role": "user", "content": user_prompt}],
        system=_DOC_CHAT_SYSTEM_PROMPT,
    )
    return response.strip()


def ask_riskgpt_for_blocks(
    selected_blocks: List[Dict[str, Any]],
    user_prompt: str,
    full_markdown: str,
    template_name: Optional[str] = None,
    template_content: Optional[str] = None,
    all_suggestions: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Ask RiskGPT to improve selected blocks or answer general questions about the document.
    
    Args:
        selected_blocks: List of block metadata dicts with 'id', 'content', 'type' (empty for general chat)
        user_prompt: User's natural language request
        full_markdown: Full document markdown for context
        template_name: Name of template being used (optional)
        template_content: Full template markdown (optional)
        all_suggestions: All pending/accepted/rejected suggestions (optional)
        conversation_history: Last 5 messages for context (optional)
    
    Returns:
        Dict with structure:
        {
            "analysis": "Explanation of what needs to be improved and why...",
            "suggestions": [...]  # Only if blocks were selected
        }
    """
    if not user_prompt.strip():
        raise ValueError("user_prompt must be provided")
    
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")
    
    client = get_llm_client()
    
    # Determine if this is block-specific or general chat
    is_block_specific = selected_blocks and len(selected_blocks) > 0
    
    # Build system prompt with persona
    system_prompt = """You are RiskGPT, a helpful document review assistant.

YOUR ROLE:
- Help users write and review policy documents
- Answer questions naturally about the document, template, or review process
- Provide guidance, explanations, and analysis
- Stay within the scope of document review

CONTEXT YOU HAVE:
- The full document being reviewed
- The template being applied (if any)
- All suggestions (pending, accepted, rejected)
- Recent conversation history
- Selected blocks (if user selected specific text)

CONVERSATION STYLE:
- Be conversational and helpful
- Answer questions directly
- Use markdown formatting for readability (headings, lists, bold, etc.)
- Reference specific sections when helpful
- Keep responses concise but complete

RULES:
1. ONLY help with document review - politely decline unrelated requests
2. Use conversation history to maintain context
3. Respond in natural markdown format
4. Be specific and cite document sections when relevant"""
    
    # Build user content with all context
    context_sections = []
    
    # Add conversation history
    if conversation_history:
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation_history[-5:]  # Last 5 messages
        ])
        context_sections.append(f"RECENT CONVERSATION:\n{history_text}")
    
    # Add current request
    context_sections.append(f"CURRENT REQUEST:\n{user_prompt}")
    
    # Add selected blocks if any
    if is_block_specific:
        blocks_for_llm = [
            {
                "block_id": b["id"],
                "content": b["content"],
                "type": b.get("type", "paragraph")
            }
            for b in selected_blocks
        ]
        context_sections.append(f"SELECTED BLOCKS:\n{json.dumps(blocks_for_llm, indent=2)}")
    
    # Add template context (full template)
    if template_name and template_content:
        context_sections.append(f"TEMPLATE ({template_name}):\n{template_content}")
    
    # Add suggestions context (full details)
    if all_suggestions:
        pending = [s for s in all_suggestions if s.get('status') == 'pending']
        accepted = [s for s in all_suggestions if s.get('status') == 'accepted']
        rejected = [s for s in all_suggestions if s.get('status') == 'rejected']
        
        suggestions_summary = f"Total suggestions: {len(all_suggestions)}\n"
        suggestions_summary += f"Pending: {len(pending)}\n"
        suggestions_summary += f"Accepted: {len(accepted)}\n"
        suggestions_summary += f"Rejected: {len(rejected)}\n\n"
        
        # Add full details of all suggestions
        if pending:
            suggestions_summary += "PENDING SUGGESTIONS:\n"
            for i, sug in enumerate(pending, 1):
                block_id = sug.get('block_id', 'unknown')
                reasoning = sug.get('reasoning', '')
                changes = sug.get('changes_made', [])
                suggestions_summary += f"{i}. Block {block_id}:\n"
                suggestions_summary += f"   Reasoning: {reasoning}\n"
                if changes:
                    suggestions_summary += f"   Changes: {', '.join(changes)}\n"
                suggestions_summary += "\n"
        
        context_sections.append(f"SUGGESTIONS:\n{suggestions_summary}")
    
    # Add full document (increased limit)
    context_sections.append(f"FULL DOCUMENT:\n{full_markdown[:20000]}")
    
    user_content = "\n\n".join(context_sections)
    
    logger.info(f"RiskGPT calling LLM with {len(user_content)} chars of context")
    
    try:
        response = client.invoke(
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt
        )
    except Exception as e:
        logger.error(f"LLM invoke failed: {e}")
        raise
    
    logger.info(f"RiskGPT response length: {len(response)} chars")
    if not response or not response.strip():
        logger.error("RiskGPT returned empty response!")
        raise ValueError("LLM returned empty response")
    
    # Return plain markdown response - no JSON parsing!
    logger.info(f"RiskGPT responded conversationally")
    return {
        "analysis": response.strip(),
        "suggestions": []
    }
