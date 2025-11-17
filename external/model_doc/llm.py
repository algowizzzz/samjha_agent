from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from external.agent.llm_client import get_llm_client, is_llm_available

logger = logging.getLogger(__name__)


class LLMNotAvailableError(RuntimeError):
    pass


def call_llm_json(system_prompt: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Call LLM with JSON payload and return parsed JSON response."""
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
        
        parsed = json.loads(cleaned_response)
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


_SUMMARIZE_FILE_SYSTEM_PROMPT = (
    "You are an expert Python code analyst. "
    "Analyze the provided Python code and its AST structure to generate a comprehensive summary. "
    "Include: purpose, key classes/functions, dependencies, and important patterns or algorithms. "
    "Be concise but thorough. Focus on what the code does, not implementation details."
)


def summarize_file(code_content: str, ast_structure: Dict[str, Any]) -> str:
    """Generate a summary for a single Python file."""
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")

    client = get_llm_client()
    
    # Prepare context
    classes = ast_structure.get("classes", [])
    functions = ast_structure.get("functions", [])
    imports = ast_structure.get("imports", [])
    
    context_parts = [
        f"Code Content:\n{code_content[:10000]}",  # Limit code to 10k chars
        f"\nAST Structure:",
        f"Classes: {len(classes)}",
        f"Functions: {len(functions)}",
        f"Imports: {len(imports)}",
    ]
    
    if classes:
        context_parts.append(f"\nClass names: {', '.join([c.get('name', '') for c in classes[:10]])}")
    if functions:
        context_parts.append(f"\nFunction names: {', '.join([f.get('name', '') for f in functions[:10]])}")
    
    user_prompt = "\n".join(context_parts)
    
    response = client.invoke(
        messages=[{"role": "user", "content": user_prompt}],
        system=_SUMMARIZE_FILE_SYSTEM_PROMPT,
    )
    return response.strip()


_GENERATE_HIERARCHICAL_SUMMARY_SYSTEM_PROMPT = (
    "You are an expert software architect. "
    "Given summaries of individual files and the codebase hierarchy, generate a comprehensive hierarchical summary. "
    "Organize by modules/packages, identify key components, dependencies, and the overall architecture. "
    "Provide both structural overview and logical relationships between components."
)


def generate_hierarchical_summary(
    file_summaries: Dict[str, str], hierarchy: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a hierarchical summary of the entire codebase."""
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")

    client = get_llm_client()
    
    # Prepare file summaries text
    summaries_text = []
    for file_path, summary in list(file_summaries.items())[:50]:  # Limit to 50 files
        summaries_text.append(f"File: {file_path}\nSummary: {summary}\n")
    
    payload = {
        "file_summaries": "\n\n".join(summaries_text),
        "hierarchy": json.dumps(hierarchy, indent=2),
    }
    
    user_prompt = json.dumps(payload, indent=2)
    
    response = client.invoke(
        messages=[{"role": "user", "content": user_prompt}],
        system=_GENERATE_HIERARCHICAL_SUMMARY_SYSTEM_PROMPT,
    )
    
    # Parse response as JSON if possible, otherwise return as text
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) > 2:
                cleaned = "\n".join(lines[1:-1])
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "component_summary": response.strip(),
            "module_hierarchy": hierarchy,
            "key_modules": [],
            "key_classes": [],
            "dependencies": [],
        }


_CREATE_DOCUMENTATION_OUTLINE_SYSTEM_PROMPT = (
    "You are a technical documentation expert. "
    "Given a documentation template structure, hierarchical codebase summary, and file summaries, "
    "generate a detailed documentation outline that maps codebase components to template sections. "
    "Return JSON with section mappings, template_path, and confidence scores."
)


def create_documentation_outline(
    template: Dict[str, Any],
    hierarchical_summary: Dict[str, Any],
    file_summaries: Dict[str, str],
) -> Dict[str, Any]:
    """Generate a documentation outline mapping codebase to template."""
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")

    # Use call_llm_json for structured output
    payload = {
        "template": template,
        "hierarchical_summary": hierarchical_summary,
        "file_summaries_count": len(file_summaries),
        "sample_file_summaries": dict(list(file_summaries.items())[:10]),
    }
    
    result = call_llm_json(_CREATE_DOCUMENTATION_OUTLINE_SYSTEM_PROMPT, payload)
    
    if result and isinstance(result[0], dict):
        return result[0]
    
    # Fallback structure
    return {
        "template_id": template.get("id", "unknown"),
        "template_structure": template,
        "section_mappings": [],
        "unmapped_sections": [],
        "outline_metadata": {},
    }


_DRAFT_SECTION_SYSTEM_PROMPT = (
    "You are a technical writer specializing in software documentation. "
    "Given a documentation section outline, relevant file summaries, and template guidance, "
    "write comprehensive, clear, and accurate documentation content. "
    "Use proper markdown formatting, code examples when appropriate, and maintain consistency with the template style."
)


def draft_section(
    outline_section: Dict[str, Any],
    relevant_summaries: List[str],
    template: Optional[Dict[str, Any]] = None,
) -> str:
    """Draft content for a single documentation section."""
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")

    client = get_llm_client()
    
    section_id = outline_section.get("section_id", "")
    section_title = outline_section.get("heading", "Section")
    template_guidance = outline_section.get("template_guidance", "")
    
    context_parts = [
        f"Section: {section_title}",
        f"Section ID: {section_id}",
    ]
    
    if template_guidance:
        context_parts.append(f"\nTemplate Guidance:\n{template_guidance}")
    
    if relevant_summaries:
        context_parts.append(f"\nRelevant Code Summaries:\n" + "\n\n".join(relevant_summaries[:5]))
    
    user_prompt = "\n".join(context_parts)
    
    response = client.invoke(
        messages=[{"role": "user", "content": user_prompt}],
        system=_DRAFT_SECTION_SYSTEM_PROMPT,
    )
    return response.strip()


_CODEBASE_CHAT_SYSTEM_PROMPT = (
    "You are an expert codebase documentation assistant. "
    "Use the provided context (codebase summary, file summaries, selected code) to answer questions about the codebase. "
    "Reference specific files, classes, or functions when relevant. "
    "If the answer is unclear from the supplied information, state what additional information is needed."
)


def generate_chat_reply(
    message: str,
    codebase_state: Dict[str, Any],
    selection: Optional[str] = None,
    max_excerpt_chars: int = 5000,
    max_selection_chars: int = 1500,
) -> str:
    """Generate a chat reply about the codebase."""
    if not message.strip():
        raise ValueError("message must be provided")

    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")

    client = get_llm_client()

    codebase_metadata = codebase_state.get("codebase_metadata", {}) or {}
    codebase_name = codebase_metadata.get("codebase_id") or codebase_state.get("codebase_id") or "Codebase"

    hierarchical_summary = codebase_state.get("hierarchical_summary", {}) or {}
    summary_text = hierarchical_summary.get("component_summary", "")[:2000]

    file_summaries = codebase_state.get("file_summaries", {}) or {}
    summary_list = []
    for file_path, summary in list(file_summaries.items())[:10]:
        summary_list.append(f"File: {file_path}\n{summary[:200]}")

    selection_snippet = selection[:max_selection_chars] if selection else ""
    final_doc = codebase_state.get("final_documentation", "") or ""
    doc_excerpt = final_doc[:max_excerpt_chars] if final_doc else ""

    user_sections = [
        f"Codebase: {codebase_name}",
        f"Question:\n{message.strip()}",
    ]
    
    if summary_text:
        user_sections.append(f"Codebase Summary:\n{summary_text}")
    
    if summary_list:
        user_sections.append(f"File Summaries:\n" + "\n\n".join(summary_list))
    
    if selection_snippet:
        user_sections.append(f"Selected Code:\n{selection_snippet}")
    
    if doc_excerpt:
        user_sections.append(f"Documentation Excerpt:\n{doc_excerpt}")

    user_prompt = "\n\n".join(user_sections)

    response = client.invoke(
        messages=[{"role": "user", "content": user_prompt}],
        system=_CODEBASE_CHAT_SYSTEM_PROMPT,
    )
    return response.strip()

