"""
Web Research Decider - LLM-only reasoning component.
Produces ResearchSpec + ResearchPlan, decides ASK_USER/EXECUTE/BLOCK.
"""

import json
import logging
import re
from pathlib import Path
import os
from typing import Dict, Any, Optional
from external.agent.state_types import ResearchControllerState

logger = logging.getLogger(__name__)

try:
    from external.platform.llm import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Decider-only Anthropic thinking/reasoning.
DECIDER_THINKING_ENABLED: bool = os.getenv("DECIDER_THINKING_ENABLED", "0") == "1"
DECIDER_THINKING_BUDGET_TOKENS: int = int(os.getenv("DECIDER_THINKING_BUDGET_TOKENS", "2000"))
DECIDER_MAX_TOKENS: int = int(os.getenv("DECIDER_MAX_TOKENS", "16000"))
THINKING_TRACE_MAX_CHARS: int = int(os.getenv("THINKING_TRACE_MAX_CHARS", "20000"))


def load_web_research_decider_prompt(agent_id: Optional[str] = None) -> str:
    """Load Web Research Decider prompt from DB or file."""
    # Try to load from DB first (if agent_id provided)
    if agent_id:
        try:
            from core.db.session import get_db_session
            from external.agent.persistence import get_prompt_content
            with get_db_session() as db:
                prompt_content = get_prompt_content(db, "web_research_decider", category="web_search")
                if prompt_content:
                    return prompt_content
        except Exception as e:
            logger.warning(f"Failed to load prompt from DB for agent {agent_id}: {e}")
    
    # Fallback to file
    prompt_path = Path("external/config/prompts/web_research_decider.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        logger.warning(f"Web Research Decider prompt not found at {prompt_path}, using fallback")
        return "# WEB RESEARCH DECIDER PROMPT\n\nOutput JSON only."


def parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response (may include markdown code blocks)."""
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith("```"):
        first_newline = response.find("\n")
        if first_newline != -1:
            response = response[first_newline + 1:]
        if response.endswith("```"):
            response = response[:-3].strip()
        elif "```" in response:
            response = response.rsplit("```", 1)[0].strip()
    
    # Remove control characters except newlines, carriage returns, and tabs
    response = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response)
    
    # Try to parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {e}")
        logger.debug(f"Response was: {response[:500]}")
        raise ValueError(f"Invalid JSON in Web Research Decider response: {e}")


def run_web_research_decider(state: ResearchControllerState) -> dict:
    """
    Single LLM call using Web Research Decider prompt.
    
    Args:
        state: Research controller state with user_query, domain_md, etc.
        
    Returns:
        Schema-valid decider_output dictionary with research_spec
        
    Raises:
        ValueError: If validation fails after 3 attempts
    """
    if not LLM_AVAILABLE:
        raise ValueError("LLM client not available for Web Research Decider")
    
    llm_client = get_llm_client()
    if not llm_client.is_available():
        raise ValueError("LLM client is not available")
    
    agent_id = state.get("agent_id")
    prompt_template = load_web_research_decider_prompt(agent_id)
    
    # Extract continuity_packet if available
    continuity_packet = state.get("continuity_packet", {})
    prior_research_spec_from_continuity = continuity_packet.get("prior_research_spec", {})
    prior_research_spec_status_from_continuity = continuity_packet.get("prior_research_spec_status", {})
    
    # Use continuity_packet values if available, otherwise use state
    prior_research_spec = prior_research_spec_from_continuity if prior_research_spec_from_continuity else state.get("research_spec", {})
    prior_research_spec_status = prior_research_spec_status_from_continuity if prior_research_spec_status_from_continuity else state.get("research_spec_status", {})
    
    # Build prompt context from state
    context = {
        "user_query": state.get("user_query", ""),
        "conversation_history": json.dumps(state.get("conversation_history", []), indent=2),
        "domain_md": state.get("domain_md", ""),
        "prior_research_spec": json.dumps(prior_research_spec, indent=2),
        "prior_research_spec_status": json.dumps(prior_research_spec_status, indent=2),
        "evidence_pack": json.dumps(state.get("evidence_pack", {}), indent=2) if state.get("evidence_pack") else "None",
        "continuity_packet": json.dumps(continuity_packet, indent=2),
        "last_executor_report": json.dumps(state.get("last_executor_report", {}), indent=2) if state.get("last_executor_report") else "None",
        "policy_limits": json.dumps(state.get("policy_limits", {}), indent=2),
        "iteration_count": state.get("iteration_count", 0),
        "sources_seen": json.dumps(state.get("sources_seen", []), indent=2),
    }
    
    # Format prompt with context
    prompt = f"""{prompt_template}

## CURRENT CONTEXT

User Query:
{context['user_query']}

Conversation History:
{context['conversation_history']}

Domain Configuration:
{context['domain_md']}

Prior Research Spec:
{context['prior_research_spec']}

Prior Research Spec Status:
{context['prior_research_spec_status']}

Evidence Pack (from prior iterations, if any):
{context['evidence_pack']}

Continuity Packet (standardized; may be empty):
{context['continuity_packet']}

Last Executor Report:
{context['last_executor_report']}

Policy Limits:
{context['policy_limits']}

Current Iteration:
{context['iteration_count']}

Sources Already Seen (to avoid duplicates):
{context['sources_seen']}

---

Output your decision as JSON only (no markdown, no prose):
"""
    
    # Try up to 3 times (initial + 2 retries)
    last_error = None
    for attempt in range(3):
        try:
            want_thinking = bool(state.get("show_thinking", False)) or DECIDER_THINKING_ENABLED
            
            # Get agent's configured model (defaults to Sonnet)
            agent_model = state.get("agent_model") or "claude-3-sonnet-20240229"
            
            # Model-specific settings
            model = agent_model
            model_max_tokens = 4096
            model_supports_thinking = False
            
            if model:
                model_lower = model.lower()
                if "sonnet" in model_lower:
                    model_max_tokens = 8192
                    model_supports_thinking = True
                elif "haiku" in model_lower:
                    model_max_tokens = 4096
                    model_supports_thinking = False
                elif "opus" in model_lower:
                    model_max_tokens = 4096
                    model_supports_thinking = True
            
            if want_thinking and not model_supports_thinking:
                logger.info(f"Model {model} does not support thinking mode. Disabling thinking.")
                want_thinking = False
            
            base_max_tokens = min(DECIDER_MAX_TOKENS, model_max_tokens)
            
            thinking = None
            temperature = 0.0
            max_tokens = base_max_tokens
            if want_thinking and DECIDER_THINKING_BUDGET_TOKENS > 0:
                thinking = {"type": "enabled", "budget_tokens": DECIDER_THINKING_BUDGET_TOKENS}
                temperature = 1.0
                required_max = int(thinking.get("budget_tokens", 0)) + 1
                max_tokens = min(max(base_max_tokens, required_max), model_max_tokens)
                
                detailed = llm_client.invoke_with_prompt_detailed(
                    system_prompt="",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format="json",
                    thinking=thinking,
                    model=agent_model
                )
                thinking_text = detailed.get("thinking") or ""
                if THINKING_TRACE_MAX_CHARS > 0 and len(thinking_text) > THINKING_TRACE_MAX_CHARS:
                    thinking_text = thinking_text[:THINKING_TRACE_MAX_CHARS] + "\n... (truncated)"
                state["thinking_trace"] = thinking_text
                response_text = detailed.get("text") or ""
            else:
                state["thinking_trace"] = None
                response_text = llm_client.invoke_with_prompt(
                    system_prompt="",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format="json",
                    thinking=None,
                    model=agent_model
                )
            
            output = parse_json_response(response_text)
            
            # Enforce RETRY query_type if executor error exists
            last_rep = state.get("last_executor_report")
            if isinstance(last_rep, dict) and last_rep:
                if last_rep.get("status") == "ERROR" and output.get("query_type") != "USER_ANSWER":
                    output["query_type"] = "RETRY"
                    sigs = output.get("query_type_signals", [])
                    if not isinstance(sigs, list):
                        sigs = []
                    if "prior executor error present" not in sigs:
                        sigs.append("prior executor error present")
                    output["query_type_signals"] = sigs
            
            # Basic validation (ensure required fields exist)
            if "action" not in output:
                raise ValueError("Missing 'action' in decider output")
            if "research_spec" not in output:
                raise ValueError("Missing 'research_spec' in decider output")
            
            logger.info(f"Web Research Decider succeeded on attempt {attempt + 1}")
            return output
            
        except Exception as e:
            last_error = e
            logger.warning(f"Web Research Decider attempt {attempt + 1} failed: {e}")
            if attempt < 2:  # 0-indexed, so attempt < 2 means we have retries left
                continue
            else:
                raise ValueError(f"Web Research Decider failed after 3 attempts: {last_error}")
    
    raise ValueError(f"Web Research Decider failed after 3 attempts: {last_error}")

