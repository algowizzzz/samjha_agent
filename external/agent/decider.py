"""
Decider - LLM-only reasoning component.
Produces QuerySpec + InvestigationPlan, decides ASK_USER/EXECUTE/BLOCK.
"""

import json
import logging
import re
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


def load_decider_prompt(agent_id: Optional[str] = None) -> str:
    """Load Decider prompt from DB (with agent override) or file."""
    # Try to load from DB first (with agent override if provided)
    if agent_id:
        try:
            from external.core.db.session import get_db_session
            from external.agent.persistence import get_prompt_content
            with get_db_session() as db:
                prompt_content = get_prompt_content(db, "decider", category="structured", agent_id=agent_id)
                if prompt_content:
                    return prompt_content
        except Exception as e:
            logger.warning(f"Failed to load prompt from DB for agent {agent_id}: {e}")
    
    # Try global DB prompt
    try:
        from external.core.db.session import get_db_session
        from external.agent.persistence import get_prompt_content
        with get_db_session() as db:
            prompt_content = get_prompt_content(db, "decider", category="structured")
            if prompt_content:
                return prompt_content
    except Exception as e:
        logger.warning(f"Failed to load prompt from DB: {e}")
    
    # Fallback to file
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
    
    # Remove control characters that can break JSON parsing (but preserve \n, \t, \r)
    # Control characters are chars < 0x20 except for \n (0x0A), \r (0x0D), \t (0x09)
    # Remove control characters except newlines, carriage returns, and tabs
    response = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response)
    
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
    
    agent_id = state.get("agent_id")
    prompt_template = load_decider_prompt(agent_id=agent_id)
    
    # Build prompt context from state
    # DIAGNOSTIC: Log what state contains before building context
    qs = state.get("query_spec", {})
    cp = state.get("continuity_packet", {})
    logger.info(f"[DATA_FLOW] Decider received state.query_spec.start_table: {qs.get('start_table', {})}")
    logger.info(f"[DATA_FLOW] Decider received state.query_spec.dimensions: {qs.get('dimensions', [])}")
    logger.info(f"[DATA_FLOW] Decider received state.query_spec keys: {list(qs.keys()) if qs else 'EMPTY'}")
    logger.info(f"[DATA_FLOW] Decider received continuity_packet.pending_clarification: {cp.get('pending_clarification', {})}")
    
    context = {
        "user_query": state.get("user_query", ""),
        "conversation_history": json.dumps(state.get("conversation_history", []), indent=2),
        "domain_md": state.get("domain_md", ""),
        "prior_query_spec": json.dumps(state.get("query_spec", {}), indent=2),
        "prior_query_spec_status": json.dumps(state.get("query_spec_status", {}), indent=2),
        "continuity_packet": json.dumps(state.get("continuity_packet", {}), indent=2),
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

Continuity Packet (standardized; may be empty):
{context['continuity_packet']}

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

            # Get agent's configured model (defaults to Sonnet)
            agent_model = state.get("agent_model") or "claude-3-sonnet-20240229"
            
            # Get model-specific max_tokens limit and thinking support
            # Haiku: 4096, NO thinking support
            # Sonnet 3.5: 8192, supports thinking
            # Sonnet 4: 8192, supports thinking
            # Opus: 4096, supports thinking (but check model version)
            model = agent_model
            model_max_tokens = 4096  # Default conservative limit
            model_supports_thinking = False  # Default: assume no thinking support
            
            if model:
                model_lower = model.lower()
                if "sonnet" in model_lower:
                    model_max_tokens = 8192
                    model_supports_thinking = True  # Sonnet models support thinking
                elif "haiku" in model_lower:
                    model_max_tokens = 4096
                    model_supports_thinking = False  # Haiku does NOT support thinking
                elif "opus" in model_lower:
                    model_max_tokens = 4096
                    model_supports_thinking = True  # Opus supports thinking
            
            # Disable thinking if model doesn't support it
            if want_thinking and not model_supports_thinking:
                logger.info(f"Model {model} does not support thinking mode. Disabling thinking.")
                want_thinking = False
            
            # Use the minimum of requested max_tokens and model limit
            base_max_tokens = min(DECIDER_MAX_TOKENS, model_max_tokens)
            logger.debug(f"[DECIDER] Model: {model}, model_max_tokens: {model_max_tokens}, base_max_tokens: {base_max_tokens}, thinking_enabled: {want_thinking}")

            thinking = None
            temperature = 0.0
            max_tokens = base_max_tokens
            if want_thinking and DECIDER_THINKING_BUDGET_TOKENS > 0:
                thinking = {"type": "enabled", "budget_tokens": DECIDER_THINKING_BUDGET_TOKENS}
                # Anthropic extended thinking requires temperature=1 and max_tokens > budget_tokens
                temperature = 1.0
                # Ensure max_tokens respects model limit AND is > budget_tokens
                required_max = int(thinking.get("budget_tokens", 0)) + 1
                max_tokens = min(max(base_max_tokens, required_max), model_max_tokens)

                detailed = llm_client.invoke_with_prompt_detailed(
                    system_prompt="",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format="json",
                    thinking=thinking,
                    model=agent_model  # Use agent's configured model
                )
                thinking_text = detailed.get("thinking") or ""
                if THINKING_TRACE_MAX_CHARS > 0 and len(thinking_text) > THINKING_TRACE_MAX_CHARS:
                    thinking_text = thinking_text[:THINKING_TRACE_MAX_CHARS] + "\n... (truncated)"
                state["thinking_trace"] = thinking_text
                response_text = detailed.get("text") or ""
            else:
                state["thinking_trace"] = None
                # Use the same model-respecting max_tokens for non-thinking path
                response_text = llm_client.invoke_with_prompt(
                    system_prompt="",
                    user_prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,  # Already set to base_max_tokens (respects model limit)
                    response_format="json",
                    thinking=None,
                    model=agent_model  # Use agent's configured model
                )

            output = parse_json_response(response_text)

            # ------------------------------------------------------------------
            # Deterministic retry semantics (guardrails)
            # ------------------------------------------------------------------
            # The LLM sometimes misclassifies retries as FOLLOW_UP/NEW_QUERY even when an executor
            # error exists. In controller retry loops, `state.last_executor_report` is the signal
            # that we're retrying the same intent due to failure.
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

                    # If the last error clearly indicates a missing column/table, prefer ASK_USER
                    # to avoid burning attempts on repeated EXECUTE.
                    last_err = (last_rep.get("last_error") or "")
                    last_err_l = str(last_err).lower()
                    missing_signal = (
                        "referenced column" in last_err_l
                        or "not found in from clause" in last_err_l
                        or "binder error" in last_err_l
                        or "does not exist" in last_err_l
                        or "table" in last_err_l and "does not exist" in last_err_l
                    )
                    if missing_signal:
                        # Try to extract the missing identifier and candidate bindings from DuckDB error text
                        import re
                        missing_name = None
                        m = re.search(r'referenced column\\s+\"([^\"]+)\"\\s+not found', last_err, flags=re.IGNORECASE)
                        if m:
                            missing_name = m.group(1)
                        # Candidate bindings (DuckDB often provides a helpful list)
                        candidates = None
                        try:
                            for line in str(last_err).splitlines():
                                if line.strip().lower().startswith("candidate bindings:"):
                                    candidates = line.split(":", 1)[1].strip()
                                    break
                        except Exception:
                            candidates = None

                        output["action"] = "ASK_USER"

                        # Prefer an LLM-crafted clarification message (more helpful UX),
                        # but fall back to a deterministic message if anything fails.
                        ask_user_payload = None
                        try:
                            from external.platform.prompt_loader import load_prompt
                            llm_client_for_ask = get_llm_client()
                            if llm_client_for_ask and llm_client_for_ask.is_available():
                                tmpl = load_prompt("ask_user_clarification")
                                dataset_path = (output.get("query_spec", {}) or {}).get("start_table", {}).get("path", "") or ""
                                prompt2 = tmpl.format(
                                    user_query=state.get("user_query", ""),
                                    last_error=str(last_err),
                                    dataset_path=dataset_path,
                                    attempt_count=str(state.get("attempt_count", 0)),
                                    max_attempts=str((state.get("policy_limits") or {}).get("max_attempts", "")),
                                    candidate_bindings=candidates or "",
                                    missing_field=missing_name or ""
                                )
                                resp_json = llm_client_for_ask.invoke_with_prompt(
                                    system_prompt="Return JSON only. Do not include markdown.",
                                    user_prompt=prompt2,
                                    temperature=0.2,
                                    max_tokens=400,
                                    response_format="json"
                                )
                                ask_user_payload = parse_json_response(resp_json)
                        except Exception:
                            ask_user_payload = None

                        if isinstance(ask_user_payload, dict) and all(k in ask_user_payload for k in ("question", "why_non_defaultable", "what_answer_unblocks")):
                            # Guardrail: if model returns a non-question or just repeats the user query, fall back.
                            qtxt = str(ask_user_payload.get("question", "")).strip()
                            uq = str(state.get("user_query", "") or "").strip()
                            if (not qtxt) or (uq and qtxt.lower() == uq.lower()):
                                ask_user_payload = None
                            elif missing_name and (missing_name.lower() not in qtxt.lower()):
                                # Missing field should be mentioned for clarity
                                ask_user_payload = None

                        if isinstance(ask_user_payload, dict) and all(k in ask_user_payload for k in ("question", "why_non_defaultable", "what_answer_unblocks")):
                            output["ask_user"] = {
                                "question": str(ask_user_payload.get("question", "")),
                                "why_non_defaultable": str(ask_user_payload.get("why_non_defaultable", "")),
                                "what_answer_unblocks": str(ask_user_payload.get("what_answer_unblocks", "")),
                            }
                            # If we have candidate bindings from the engine and the LLM didn't mention
                            # any, append a short hint so the user can answer quickly.
                            try:
                                if candidates:
                                    q_txt = output["ask_user"]["question"] or ""
                                    q_low = q_txt.lower()
                                    # If the question doesn't already contain a hint, append candidate columns
                                    # so the user can answer quickly without guessing.
                                    if "candidate columns" in q_low:
                                        raise RuntimeError("already has candidate hint")
                                    cand_tokens = [c.strip().strip('"').strip("'") for c in str(candidates).split(",") if c.strip()]
                                    if cand_tokens:
                                        output["ask_user"]["question"] = (
                                            q_txt.rstrip()
                                            + f" (Candidate columns: {', '.join(cand_tokens[:5])})"
                                        )
                            except Exception:
                                pass
                        else:
                            # Deterministic fallback
                            q_missing = f"`{missing_name}`" if missing_name else "that field"
                            cand_txt = f" Available columns include: {candidates}." if candidates else ""
                            output["ask_user"] = {
                                "question": f"I can’t find {q_missing} in this dataset.{cand_txt} Which column should I use instead?",
                                "why_non_defaultable": "Choosing the wrong column would change the meaning of the results.",
                                "what_answer_unblocks": "Tell me the correct column name to use (or what concept you mean), and I’ll rerun the query."
                            }
                        # Align decision fields if present
                        if isinstance(output.get("decisions"), dict):
                            output["decisions"]["clarification_need"] = "ASK_REQUIRED"
                            output["decisions"]["determinacy"] = "UNDERDETERMINED"
            
            # LOG FOR COMPARISON: thinking=true vs thinking=false
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

            # Coherence: if business_question is present in query_spec, it must not be treated as missing.
            # This prevents the SQL gate from blocking on business_question due to missing status defaults.
            try:
                bq = (output.get("query_spec", {}) or {}).get("business_question", "")
                if isinstance(bq, str) and bq.strip():
                    bqs = output.get("query_spec_status", {}).get("business_question", {})
                    if bqs.get("status") in ["missing", "", None]:
                        bqs["status"] = "inferred"
                        # Source enum must be one of: domain_md | tool_result | user | rule
                        bqs["source"] = bqs.get("source") if bqs.get("source") in ["domain_md", "tool_result", "user", "rule"] else "user"
                        bqs["notes"] = bqs.get("notes") or "Derived from user query"
                        bqs["blocks_execution"] = False
                        output["query_spec_status"]["business_question"] = bqs
            except Exception:
                pass

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
                
                # ------------------------------------------------------------------
                # Post-process: Route ALL ASK_USER cases through ask_user_clarification.md
                # This ensures consistent, helpful clarification messages with full context.
                # ------------------------------------------------------------------
                if output.get("action") == "ASK_USER":
                    # Determine what was found vs what's missing from query_spec and status
                    qs = output.get("query_spec", {})
                    qss = output.get("query_spec_status", {})
                    
                    # Build "found" summary
                    found_items = []
                    if qs.get("business_question"):
                        found_items.append(f"Business question: {qs['business_question']}")
                    if qs.get("start_table", {}).get("path"):
                        found_items.append(f"Table: {qs['start_table']['path']}")
                    if qs.get("dimensions"):
                        found_items.append(f"Dimensions: {', '.join(qs['dimensions'])}")
                    if qs.get("metrics"):
                        metric_names = [m.get("name") if isinstance(m, dict) else str(m) for m in qs["metrics"]]
                        found_items.append(f"Metrics: {', '.join(metric_names)}")
                    query_spec_found = "\n".join(found_items) if found_items else "None yet"
                    
                    # Build "missing" summary from status and track the primary blocking field
                    missing_items = []
                    blocking_field = None  # Track the first/primary blocking field for USER_ANSWER
                    for field, status_obj in (qss or {}).items():
                        if isinstance(status_obj, dict):
                            st = status_obj.get("status", "")
                            be = status_obj.get("blocks_execution", False)
                            if st in ("missing", "conflict") or be:
                                missing_items.append(f"- {field}: {status_obj.get('notes', 'missing')}")
                                # Track the first blocking field for fills_gap
                                if blocking_field is None:
                                    blocking_field = field
                    query_spec_missing = "\n".join(missing_items) if missing_items else "No blocking issues detected (this may be a comprehension/clarity issue)"
                    
                    # Extract missing field and candidates if available (for retry cases)
                    missing_name = None
                    candidates = None
                    last_err = ""
                    if isinstance(state.get("last_executor_report"), dict):
                        last_rep = state.get("last_executor_report", {})
                        last_err = str(last_rep.get("last_error", ""))
                        if last_err:
                            import re
                            m = re.search(r'referenced column\s+\"([^\"]+)\"\s+not found', last_err, flags=re.IGNORECASE)
                            if m:
                                missing_name = m.group(1)
                            for line in last_err.splitlines():
                                if line.strip().lower().startswith("candidate bindings:"):
                                    candidates = line.split(":", 1)[1].strip()
                                    break
                    
                    # Build conversation history summary
                    conv_hist = state.get("conversation_history", [])
                    conv_summary = ""
                    if conv_hist:
                        conv_summary = "\n".join([
                            f"Turn {i+1}: User: {t.get('query', '')[:100]}"
                            for i, t in enumerate(conv_hist[-3:])  # Last 3 turns
                        ])
                    else:
                        conv_summary = "None (first query in conversation)"
                    
                    # Get comprehension/determinacy from decisions
                    decisions = output.get("decisions", {})
                    comprehension_status = decisions.get("comprehension", "INTELLIGIBLE")
                    determinacy_status = decisions.get("determinacy", "DETERMINED")
                    
                    # Call ask_user_clarification.md with full context
                    try:
                        from external.platform.prompt_loader import load_prompt
                        llm_client_for_ask = get_llm_client()
                        if llm_client_for_ask and llm_client_for_ask.is_available():
                            tmpl = load_prompt("ask_user_clarification")
                            dataset_path = qs.get("start_table", {}).get("path", "") or ""
                            domain_md = state.get("domain_md", "") or ""
                            prompt2 = tmpl.format(
                                user_query=state.get("user_query", ""),
                                query_spec_found=query_spec_found,
                                query_spec_missing=query_spec_missing,
                                conversation_history=conv_summary,
                                last_error=last_err if last_err else "None (initial comprehension/determinacy check)",
                                dataset_path=dataset_path,
                                attempt_count=str(state.get("attempt_count", 0)),
                                max_attempts=str((state.get("policy_limits") or {}).get("max_attempts", "")),
                                candidate_bindings=candidates or "",
                                missing_field=missing_name or "",
                                comprehension_status=comprehension_status,
                                determinacy_status=determinacy_status,
                                domain_md=domain_md
                            )
                            resp_json = llm_client_for_ask.invoke_with_prompt(
                                system_prompt="Return JSON only. Do not include markdown.",
                                user_prompt=prompt2,
                                temperature=0.2,
                                max_tokens=500,
                                response_format="json"
                            )
                            ask_user_payload = parse_json_response(resp_json)
                            
                            # Validate the response has required fields
                            if isinstance(ask_user_payload, dict) and all(k in ask_user_payload for k in ("question", "why_non_defaultable", "what_answer_unblocks")):
                                qtxt = str(ask_user_payload.get("question", "")).strip()
                                uq = str(state.get("user_query", "") or "").strip()
                                # Guardrail: don't just repeat the user query
                                if qtxt and (not uq or qtxt.lower() != uq.lower()):
                                    output["ask_user"] = {
                                        "question": qtxt,
                                        "why_non_defaultable": str(ask_user_payload.get("why_non_defaultable", "")),
                                        "what_answer_unblocks": str(ask_user_payload.get("what_answer_unblocks", "")),
                                    }
                                    # Track which field this ASK_USER is trying to fill
                                    # This helps USER_ANSWER know which query_spec field to update
                                    if blocking_field:
                                        output["ask_user"]["fills_gap"] = blocking_field
                                    # Append candidate bindings if available and not already mentioned
                                    if candidates:
                                        q_low = qtxt.lower()
                                        if "candidate" not in q_low and "column" in q_low:
                                            cand_tokens = [c.strip().strip('"').strip("'") for c in str(candidates).split(",") if c.strip()]
                                            if cand_tokens:
                                                output["ask_user"]["question"] = (
                                                    qtxt.rstrip()
                                                    + f" (Candidate columns: {', '.join(cand_tokens[:5])})"
                                                )
                                    logger.info(f"Enhanced ASK_USER clarification via ask_user_clarification.md prompt")
                    except Exception as e:
                        logger.warning(f"Failed to enhance ASK_USER via ask_user_clarification.md: {e}. Using Decider's original ask_user.")
                        # Keep the Decider's original ask_user (or ensure it exists)
                        if not output.get("ask_user") or not output["ask_user"].get("question"):
                            output["ask_user"] = {
                                "question": "I need more information to proceed. Could you clarify your query?",
                                "why_non_defaultable": "The query requires additional details to execute.",
                                "what_answer_unblocks": "Providing the missing information will allow me to proceed."
                            }
                    
                    # Always ensure fills_gap is set if we know the blocking field
                    if blocking_field and "fills_gap" not in output.get("ask_user", {}):
                        output["ask_user"]["fills_gap"] = blocking_field
                
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

