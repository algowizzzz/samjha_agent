"""
Decider - LLM-only reasoning component.
Produces QuerySpec + InvestigationPlan, decides ASK_USER/EXECUTE/BLOCK.
"""

import json
import logging
from pathlib import Path
import os
from typing import Dict, Any, Optional
from external.agent.schema_validators import validate_decider_output
from external.agent.state_types import ControllerState

logger = logging.getLogger(__name__)

try:
    from external.platform.llm import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Decider-only Anthropic thinking/reasoning.
# Keep scoped to Decider to avoid breaking strict JSON parsing elsewhere.
DECIDER_THINKING_ENABLED: bool = os.getenv("DECIDER_THINKING_ENABLED", "0") == "1"
DECIDER_THINKING_BUDGET_TOKENS: int = int(os.getenv("DECIDER_THINKING_BUDGET_TOKENS", "2000"))
# Anthropic requirement: max_tokens must be > thinking.budget_tokens when thinking is enabled.
DECIDER_MAX_TOKENS: int = int(os.getenv("DECIDER_MAX_TOKENS", "16000"))
THINKING_TRACE_MAX_CHARS: int = int(os.getenv("THINKING_TRACE_MAX_CHARS", "20000"))


def load_decider_prompt() -> str:
    """Load Decider prompt from file."""
    prompt_path = Path("external/config/prompts/decider.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        logger.warning(f"Decider prompt not found at {prompt_path}, using fallback")
        return "# DECIDER PROMPT\n\nOutput JSON only."


def parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response (may include markdown code blocks)."""
    response = response.strip()
    
    # Remove markdown code blocks if present
    if response.startswith("```"):
        # Find the first newline after ```
        first_newline = response.find("\n")
        if first_newline != -1:
            response = response[first_newline + 1:]
        # Remove trailing ```
        if response.endswith("```"):
            response = response[:-3].strip()
        elif "```" in response:
            response = response.rsplit("```", 1)[0].strip()
    
    # Try to parse JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {e}")
        logger.debug(f"Response was: {response[:500]}")
        raise ValueError(f"Invalid JSON in Decider response: {e}")


def run_decider(state: ControllerState) -> dict:
    """
    Single LLM call using Decider prompt.
    Must validate output against decider_output.schema.json.
    If schema invalid, re-prompt internally (max 2 retries).
    
    Args:
        state: Controller state with user_query, domain_md, etc.
        
    Returns:
        Schema-valid decider_output dictionary
        
    Raises:
        ValueError: If validation fails after 3 attempts
    """
    if not LLM_AVAILABLE:
        raise ValueError("LLM client not available for Decider")
    
    llm_client = get_llm_client()
    if not llm_client.is_available():
        raise ValueError("LLM client is not available")
    
    prompt_template = load_decider_prompt()
    
    # Build prompt context from state
    context = {
        "user_query": state.get("user_query", ""),
        "conversation_history": json.dumps(state.get("conversation_history", []), indent=2),
        "domain_md": state.get("domain_md", ""),
        "prior_query_spec": json.dumps(state.get("query_spec", {}), indent=2),
        "prior_query_spec_status": json.dumps(state.get("query_spec_status", {}), indent=2),
        "last_executor_report": json.dumps(state.get("last_executor_report", {}), indent=2) if state.get("last_executor_report") else "None",
        "policy_limits": json.dumps(state.get("policy_limits", {}), indent=2)
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

Prior Query Spec:
{context['prior_query_spec']}

Prior Query Spec Status:
{context['prior_query_spec_status']}

Last Executor Report:
{context['last_executor_report']}

Policy Limits:
{context['policy_limits']}

---

Output your decision as JSON only (no markdown, no prose):
"""
    
    # Try up to 3 times (initial + 2 retries)
    last_error = None
    for attempt in range(3):
        try:
            # Use invoke_with_prompt with JSON response format
            # Per-request enablement: user/UI toggle OR env default.
            want_thinking = bool(state.get("show_thinking", False)) or DECIDER_THINKING_ENABLED

            thinking = None
            temperature = 0.0
            max_tokens = DECIDER_MAX_TOKENS
            if want_thinking and DECIDER_THINKING_BUDGET_TOKENS > 0:
                thinking = {"type": "enabled", "budget_tokens": DECIDER_THINKING_BUDGET_TOKENS}
                # Anthropic extended thinking requires temperature=1 and max_tokens > budget_tokens
                temperature = 1.0
                max_tokens = max(max_tokens, int(thinking.get("budget_tokens", 0)) + 1)

                detailed = llm_client.invoke_with_prompt_detailed(
                    system_prompt="",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format="json",
                    thinking=thinking
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
                    thinking=None
                )

            output = parse_json_response(response_text)
            
            # LOG FOR COMPARISON: thinking=true vs thinking=false
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[DECIDER_COMPARISON] thinking={want_thinking}, response_text_length={len(response_text)}")
            logger.info(f"[DECIDER_COMPARISON] response_text_snippet: {response_text[:800]}")
            grain_value = output.get('query_spec', {}).get('grain', 'MISSING')
            start_table_grain_status = output.get('query_spec_status', {}).get('start_table_grain', {})
            logger.info(f"[DECIDER_COMPARISON] grain={grain_value}, start_table_grain.status={start_table_grain_status.get('status', 'MISSING')}, blocks_execution={start_table_grain_status.get('blocks_execution', 'MISSING')}")
            
            # Post-process: Ensure ask_user is always present (required by schema)
            if "ask_user" not in output or output.get("ask_user") is None:
                output["ask_user"] = {
                    "question": "",
                    "why_non_defaultable": "",
                    "what_answer_unblocks": ""
                }
            # Also ensure block_reason is present
            if "block_reason" not in output:
                output["block_reason"] = ""
            
            # Post-process: Ensure all required query_spec fields are present
            if "query_spec" not in output:
                output["query_spec"] = {}
            
            required_spec_fields = {
                "business_question": "",
                "output_shape": {"type": "", "columns": []},
                "start_table": {"name": "", "path": ""},
                "grain": "",
                "time": {"column": "", "rule": ""},  # n_days, start, end are optional
                "metrics": [],
                "dimensions": [],
                "filters": [],
                "joins": [],
                "aggregation_plan": "",
                "validation_checks": [],
                "performance_guardrails": [],
                "defaults_used": [],
                "open_questions": []
            }
            
            for field, default_value in required_spec_fields.items():
                if field not in output["query_spec"]:
                    output["query_spec"][field] = default_value
                elif isinstance(default_value, dict) and isinstance(output["query_spec"].get(field), dict):
                    # For nested objects, ensure all required sub-fields exist
                    for sub_field, sub_default in default_value.items():
                        if sub_field not in output["query_spec"][field]:
                            output["query_spec"][field][sub_field] = sub_default
            
            # Special handling: if start_table.path exists but name is missing, derive name from path
            start_table = output["query_spec"].get("start_table", {})
            if start_table.get("path") and not start_table.get("name"):
                from pathlib import Path as P
                start_table["name"] = P(start_table["path"]).stem
            
            # Post-process: Ensure all required query_spec_status fields are present
            required_status_fields = [
                "business_question", "output_shape", "start_table_grain", "time",
                "metrics", "dimensions", "filters", "joins", "aggregation_plan",
                "validation_checks", "performance_guardrails"
            ]
            
            if "query_spec_status" not in output:
                output["query_spec_status"] = {}
            
            for field in required_status_fields:
                if field not in output["query_spec_status"]:
                    output["query_spec_status"][field] = {
                        "status": "missing",
                        "source": "rule",
                        "notes": "Not provided by decider",
                        "blocks_execution": False
                    }
                else:
                    # Ensure blocks_execution is coherent: only "missing/conflict" should block.
                    st = (output["query_spec_status"].get(field) or {}).get("status")
                    if st in ["verified", "inferred", "defaulted"]:
                        output["query_spec_status"][field]["blocks_execution"] = False

            # Coherence: if time.rule is no_time, time must not block execution.
            try:
                if (output.get("query_spec", {}).get("time", {}) or {}).get("rule") == "no_time":
                    if "time" in output.get("query_spec_status", {}):
                        output["query_spec_status"]["time"]["blocks_execution"] = False
            except Exception:
                pass
            
            # Also ensure domain, intent, and decisions are present
            if "domain" not in output:
                output["domain"] = ""
            if "intent" not in output:
                output["intent"] = ""
            if "decisions" not in output:
                output["decisions"] = {
                    "comprehension": "INTELLIGIBLE",
                    "determinacy": "DETERMINED",
                    "clarification_need": "DEFAULT_OK"
                }
            
            # Ensure investigation_plan, expected_output, stop_conditions are present
            if "investigation_plan" not in output:
                output["investigation_plan"] = []
            if "expected_output" not in output:
                output["expected_output"] = ""
            elif not isinstance(output.get("expected_output"), str):
                # Convert to string if it's not already
                output["expected_output"] = str(output.get("expected_output"))
            if "stop_conditions" not in output:
                output["stop_conditions"] = []
            
            # Post-process: Ensure query_type and query_type_signals are present (new follow-up detection)
            if "query_type" not in output:
                # Default to NEW_QUERY if not specified
                output["query_type"] = "NEW_QUERY"
            if "query_type_signals" not in output:
                output["query_type_signals"] = []
            
            # Validate output
            valid, error = validate_decider_output(output)
            if valid:
                logger.info(f"Decider output validated successfully (attempt {attempt + 1})")
                return output
            
            # Validation failed - add error to prompt and retry
            last_error = error
            logger.warning(f"Decider output validation failed (attempt {attempt + 1}): {error}")
            
            # Add validation error to prompt for next attempt
            prompt += f"\n\nVALIDATION ERROR (attempt {attempt + 1}): {error}\n\nPlease fix the JSON and output again:\n"
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"Decider execution failed (attempt {attempt + 1}): {e}")
            if attempt < 2:
                prompt += f"\n\nERROR: {e}\n\nPlease try again with valid JSON:\n"
    
    # All attempts failed
    raise ValueError(f"Decider output failed validation after 3 attempts. Last error: {last_error}")

