"""
Web Research Agent v1.0 - Decider/Executor Architecture
Controller orchestrates Decider -> Executor -> iterative loop for deep research
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from external.agent.state_types import ResearchControllerState, ResearchExecutorState
from tools.tools_registry import ToolsRegistry

logger = logging.getLogger(__name__)


def load_domain_content(user_query: str, conversation_history: list, agent_id: Optional[str] = None) -> str:
    """
    Load domain markdown content from database, or fallback to query-based selection.
    For web research agents, this contains authority domains, search scope, and research depth settings.
    """
    # If agent_id provided, load from database
    if agent_id:
        try:
            from core.db.session import get_db_session
            from external.agent.persistence import get_agent_db
            
            with get_db_session() as db:
                agent = get_agent_db(db, agent_id)
                if agent:
                    domain_content = agent.get("domain_content")
                    if domain_content:
                        return domain_content
                    else:
                        logger.warning(f"Domain content not found for agent {agent_id}")
        except Exception as e:
            logger.warning(f"Error loading domain content for agent {agent_id}: {e}")
    
    # Fallback: return empty string (web research agents require domain config)
    logger.warning("No domain content available - web research agents require domain configuration")
    return ""


def initialize_research_controller_state(
    user_query: str,
    conversation_history: list,
    prior_state: Optional[ResearchControllerState] = None,
    policy_limits: Optional[dict] = None,
    show_thinking: bool = False,
    agent_id: Optional[str] = None,
    continuity_packet: Optional[dict] = None,
) -> ResearchControllerState:
    """
    Initialize or restore research controller state.
    
    Limits conversation_history to last 5 turns to prevent prompt bloat.
    """
    # Limit conversation_history to last 5 turns
    MAX_CONVERSATION_HISTORY = 5
    if isinstance(conversation_history, list) and len(conversation_history) > MAX_CONVERSATION_HISTORY:
        conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY:]
        logger.info(f"Limited conversation_history to last {MAX_CONVERSATION_HISTORY} turns")
    
    if prior_state:
        # Restore from prior state but UPDATE user_query to the new query
        prior_state["user_query"] = user_query
        prior_state["conversation_history"] = conversation_history
        prior_state["show_thinking"] = bool(show_thinking)
        prior_state["thinking_trace"] = None
        # Reset attempt_count and last_executor_report for new query
        prior_state["attempt_count"] = 0
        prior_state["last_executor_report"] = None
        # Reset iteration_count for new query
        prior_state["iteration_count"] = 0
        return prior_state
    
    # Default policy limits for web research
    default_policy_limits = {
        "max_attempts": 3,
        "max_iterations": 4,  # Deep research: 2-4 iterations
        "max_sources": 50,
        "max_api_calls": 20,
        "timeout_seconds": 60,
    }
    
    if policy_limits:
        default_policy_limits.update(policy_limits)
    
    # Load agent config if agent_id provided
    tavily_api_key = None
    if agent_id:
        try:
            from core.db.session import get_db_session
            from external.agent.persistence import get_agent_db
            with get_db_session() as db:
                agent = get_agent_db(db, agent_id)
                if agent:
                    tavily_api_key = agent.get("tavily_api_key")  # Encrypted, will be decrypted when needed
        except Exception as e:
            logger.warning(f"Error loading agent config for {agent_id}: {e}")
    
    # Extract prior state from continuity_packet if available
    prior_research_spec = {}
    prior_research_spec_status = {}
    if continuity_packet:
        prior_research_spec = continuity_packet.get("prior_research_spec", {})
        prior_research_spec_status = continuity_packet.get("prior_research_spec_status", {})
    elif prior_state:
        prior_research_spec = prior_state.get("research_spec", {})
        prior_research_spec_status = prior_state.get("research_spec_status", {})
    
    return {
        "user_query": user_query,
        "conversation_history": conversation_history,
        "domain_md": load_domain_content(user_query, conversation_history, agent_id),
        "policy_limits": default_policy_limits,
        "research_spec": prior_research_spec,  # Initialize with prior if available
        "research_spec_status": prior_research_spec_status,  # Initialize with prior if available
        "evidence_pack": prior_state.get("evidence_pack") if prior_state else None,
        "sources_seen": prior_state.get("sources_seen", []) if prior_state else [],
        "iteration_count": prior_state.get("iteration_count", 0) if prior_state else 0,
        "show_thinking": bool(show_thinking),
        "thinking_trace": None,
        "last_executor_report": prior_state.get("last_executor_report") if prior_state else None,
        "attempt_count": 0,
        "agent_id": agent_id,
        "tavily_api_key": tavily_api_key,
        "continuity_packet": continuity_packet or {},  # Store continuity_packet for decider
    }


def handle_web_research_query(
    user_query: str,
    conversation_history: list,
    prior_state: Optional[ResearchControllerState] = None,
    tools_registry: Optional[ToolsRegistry] = None,
    policy_limits: Optional[dict] = None,
    show_thinking: bool = False,
    agent_id: Optional[str] = None,
    on_decider_output: Optional[Callable[[dict, ResearchControllerState], None]] = None,
    continuity_packet: Optional[dict] = None,
) -> dict:
    """
    Controller orchestrates:
      Decider -> Executor -> (iterative loop) until SUCCESS / ASK_USER / BLOCK / max_iterations.
    
    Args:
        user_query: User's query string
        conversation_history: List of previous conversation turns
        prior_state: Optional prior controller state (for resuming)
        tools_registry: Tools registry instance (creates default if None)
        policy_limits: Optional policy limits override
        show_thinking: Whether to show thinking trace
        agent_id: Agent instance ID
        on_decider_output: Optional callback for persisting decider output
        
    Returns:
        Response dictionary with status and results
    """
    # Initialize tools registry if not provided
    if tools_registry is None:
        tools_registry = ToolsRegistry()
        # Load Tavily tools (they should already be registered, but ensure they're available)
        # Tavily tools are registered via tools_registry.load_tools_from_config()
    
    # Initialize controller state
    state = initialize_research_controller_state(
        user_query, conversation_history, prior_state, policy_limits,
        show_thinking=show_thinking, agent_id=agent_id,
        continuity_packet=continuity_packet
    )
    max_attempts = int(state["policy_limits"]["max_attempts"])
    max_iterations = int(state["policy_limits"].get("max_iterations", 4))
    domain_md = state["domain_md"]
    
    logger.info(f"Starting web research query handling: query='{user_query[:50]}...', max_attempts={max_attempts}, max_iterations={max_iterations}")
    
    # Main loop — controller-owned retries and iterations
    while True:
        # Check max_iterations (for deep research iterations)
        if state["iteration_count"] >= max_iterations:
            logger.info(f"Max iterations ({max_iterations}) reached, synthesizing final answer")
            # Synthesize with available evidence
            return render_success_from_evidence(state)
        
        # Enforce max_attempts BEFORE calling executor
        if state["attempt_count"] >= max_attempts and state["last_executor_report"] is not None:
            logger.warning(f"Max attempts ({max_attempts}) reached - converting to ASK_USER for recovery")
            return render_error_max_attempts(
                state["last_executor_report"],
                state["attempt_count"],
                max_attempts,
                state  # Pass state for research_spec context
            )
        
        # Call Web Research Decider
        try:
            from external.agent.web_research_decider import run_web_research_decider
            decider_output = run_web_research_decider(state)  # schema-valid decider_output
        except Exception as e:
            logger.error(f"Web Research Decider failed: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "reason": f"Decider failed: {str(e)}",
                "error_type": "DECIDER_ERROR"
            }
        
        action = decider_output.get("action")
        query_type = decider_output.get("query_type", "NEW_QUERY")
        logger.info(f"Decider action: {action}, query_type: {query_type}, iteration: {state['iteration_count']}")
        
        # Get new spec from Decider
        new_research_spec = decider_output.get("research_spec", {})
        new_research_spec_status = decider_output.get("research_spec_status", {})
        
        # For FOLLOW_UP and RETRY queries, preserve missing fields from prior state
        if query_type in ("FOLLOW_UP", "RETRY") and state.get("research_spec"):
            prior_spec = state["research_spec"]
            prior_status = state.get("research_spec_status", {})
            
            # Preserve user_question if empty
            if not new_research_spec.get("user_question") and prior_spec.get("user_question"):
                if query_type == "FOLLOW_UP":
                    new_research_spec["user_question"] = f"Follow-up: {state['user_query']} (based on: {prior_spec['user_question']})"
                else:
                    new_research_spec["user_question"] = prior_spec["user_question"]
            
            # Preserve verified status fields from prior
            for field in ["user_question", "intent_type", "scope", "quality_bar", "constraints"]:
                prior_field_status = prior_status.get(field, {})
                new_field_status = new_research_spec_status.get(field, {})
                if prior_field_status.get("status") == "verified" and new_field_status.get("status") in ["missing", "", None]:
                    new_research_spec_status[field] = prior_field_status.copy()
                    new_research_spec_status[field]["notes"] = "Preserved from prior query"
        
        # Keep canonical spec in controller state
        state["research_spec"] = new_research_spec
        state["research_spec_status"] = new_research_spec_status
        
        # Optional hook for callers to persist decider output
        if on_decider_output is not None:
            try:
                enriched = dict(decider_output or {})
                enriched["research_spec"] = new_research_spec
                enriched["research_spec_status"] = new_research_spec_status
                on_decider_output(enriched, state)
            except Exception as e:
                logger.warning(f"on_decider_output hook failed: {e}", exc_info=True)
        
        # Route actions
        if action == "ASK_USER":
            logger.info("Decider requested ASK_USER")
            return render_ask_user(decider_output)
        
        if action == "BLOCK":
            logger.info("Decider requested BLOCK")
            return render_block(decider_output)
        
        if action != "EXECUTE":
            logger.error(f"Invalid action from Decider: {action}")
            return {
                "status": "ERROR",
                "reason": f"Invalid action from Decider: {action}",
                "error_type": "INVALID_ACTION"
            }
        
        # Execute once (increments attempt_count and iteration_count)
        state["attempt_count"] += 1
        state["iteration_count"] += 1
        logger.info(f"Executing attempt {state['attempt_count']}/{max_attempts}, iteration {state['iteration_count']}/{max_iterations}")
        
        try:
            from external.agent.web_research_executor_graph import run_web_research_executor
            executor_report = run_web_research_executor(decider_output, state, tools_registry, domain_md)
            state["last_executor_report"] = executor_report
            
            # Merge evidence_pack from executor into controller state
            if executor_report.get("evidence_pack"):
                # Merge with prior evidence_pack
                prior_evidence = state.get("evidence_pack") or {"sources": [], "claims": [], "conflicts": [], "gaps": []}
                new_evidence = executor_report["evidence_pack"]
                
                # Merge sources (deduplicate by URL)
                merged_sources = prior_evidence.get("sources", [])
                for source in new_evidence.get("sources", []):
                    if source.get("url") not in [s.get("url") for s in merged_sources]:
                        merged_sources.append(source)
                        state["sources_seen"].append(source.get("url"))
                
                # Merge claims
                merged_claims = prior_evidence.get("claims", []) + new_evidence.get("claims", [])
                
                # Merge conflicts
                merged_conflicts = prior_evidence.get("conflicts", []) + new_evidence.get("conflicts", [])
                
                # Merge gaps
                merged_gaps = prior_evidence.get("gaps", []) + new_evidence.get("gaps", [])
                
                state["evidence_pack"] = {
                    "sources": merged_sources,
                    "claims": merged_claims,
                    "conflicts": merged_conflicts,
                    "gaps": merged_gaps,
                }
        except Exception as e:
            logger.error(f"Executor failed: {e}", exc_info=True)
            executor_report = {
                "status": "ERROR",
                "error_type": "EXECUTOR_ERROR",
                "last_error": str(e)
            }
            state["last_executor_report"] = executor_report
        
        # Interpret executor outcome
        if executor_report.get("status") == "SUCCESS":
            # Check if we need another iteration (conflicts, gaps)
            evidence_pack = state.get("evidence_pack")
            if evidence_pack:
                conflicts = evidence_pack.get("conflicts", [])
                gaps = evidence_pack.get("gaps", [])
                
                # Check for high-severity conflicts or critical gaps
                high_severity_conflicts = [c for c in conflicts if c.get("severity") == "high"]
                critical_gaps = [g for g in gaps if g.get("criticality") == "high"]
                
                if high_severity_conflicts or critical_gaps:
                    logger.info(f"High-severity conflicts or critical gaps detected, continuing to next iteration")
                    # Continue loop - Decider will see evidence_pack and plan verification
                    continue
            
            # No conflicts/gaps or low severity - synthesize final answer
            logger.info("Executor returned SUCCESS, synthesizing final answer")
            return render_success_from_evidence(state)
        
        # executor_report["status"] == "ERROR" => controller loops back to Decider
        logger.info(f"Executor returned ERROR: {executor_report.get('error_type')}, retrying...")
        continue


def render_ask_user(decider_output: dict) -> dict:
    """Render ASK_USER response."""
    ask_user = decider_output.get("ask_user", {})
    return {
        "status": "ASK_USER",
        "question": ask_user.get("question", ""),
        "why_non_defaultable": ask_user.get("why_non_defaultable", ""),
        "what_answer_unblocks": ask_user.get("what_answer_unblocks", ""),
        "ask_user": ask_user,
    }


def render_block(decider_output: dict) -> dict:
    """Render BLOCK response."""
    return {
        "status": "BLOCK",
        "reason": decider_output.get("block_reason", "Blocked"),
        "block_reason": decider_output.get("block_reason", ""),
    }


def render_success_from_evidence(state: ResearchControllerState) -> dict:
    """Render SUCCESS response from accumulated evidence."""
    evidence_pack = state.get("evidence_pack") or {"sources": [], "claims": [], "conflicts": [], "gaps": []}
    research_spec = state.get("research_spec", {})
    
    # Get agent model
    agent_model = None
    agent_id = state.get("agent_id")
    if agent_id:
        try:
            from core.db.session import get_db_session
            from external.agent.persistence import get_agent_db
            with get_db_session() as db:
                agent = get_agent_db(db, agent_id)
                if agent:
                    agent_model = agent.get("model") or "claude-3-sonnet-20240229"
        except Exception as e:
            logger.warning(f"Error loading agent model for synthesis: {e}")
    
    # Call synthesis prompt to generate final answer
    try:
        from external.agent.web_research_synthesis import synthesize_final_answer
        final_answer = synthesize_final_answer(
            user_query=state["user_query"],
            research_spec=research_spec,
            evidence_pack=evidence_pack,
            conversation_history=state.get("conversation_history", []),
            agent_id=agent_id,
            agent_model=agent_model
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        # Fallback: simple summary
        final_answer = f"Research completed. Found {len(evidence_pack.get('sources', []))} sources with {len(evidence_pack.get('claims', []))} claims."
    
    return {
        "status": "SUCCESS",
        "final_answer": final_answer,
        "evidence_pack": evidence_pack,
        "sources": evidence_pack.get("sources", []),
        "claims": evidence_pack.get("claims", []),
        "conflicts": evidence_pack.get("conflicts", []),
        "gaps": evidence_pack.get("gaps", []),
        "iteration_count": state.get("iteration_count", 0),
        "result_summary": final_answer,  # For compatibility with existing UI
    }


def render_error_max_attempts(
    last_executor_report: dict,
    attempt_count: int,
    max_attempts: int,
    state: dict = None
) -> dict:
    """
    Convert max attempts to ASK_USER with helpful context for web research.
    
    Instead of returning a dead-end ERROR, we return ASK_USER so the user
    can rephrase or provide more details to help the agent succeed.
    """
    # Build summary of what went wrong
    attempts_summary = []
    if last_executor_report:
        last_error = last_executor_report.get("last_error", "") or last_executor_report.get("error", "")
        error_type = last_executor_report.get("error_type", "")
        
        if error_type:
            attempts_summary.append(f"Issue: {error_type}")
        if last_error:
            error_snippet = str(last_error)[:150]
            if len(str(last_error)) > 150:
                error_snippet += "..."
            attempts_summary.append(f"Details: {error_snippet}")
    
    summary_text = " | ".join(attempts_summary) if attempts_summary else "encountered issues during research"
    
    question = (
        f"I tried {attempt_count} times but couldn't complete your research query ({summary_text}). "
        f"Could you rephrase your question or provide more specific details about what you're looking for?"
    )
    
    return {
        "status": "ASK_USER",  # Changed from ERROR - give user a chance to help
        "question": question,
        "why_non_defaultable": f"After {attempt_count} attempts, I need your help to clarify the research query.",
        "what_answer_unblocks": "Rephrasing the query or providing more specific search terms would help me succeed.",
        "research_spec": state.get("research_spec", {}) if state else {},
        "research_spec_status": state.get("research_spec_status", {}) if state else {},
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "is_max_attempts_recovery": True  # Flag for UI to display appropriately
    }

