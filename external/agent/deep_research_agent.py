"""
Deep Research Agent v1.0 - Wrapper for Open Deep Research
Integrates LangChain's Open Deep Research as a new agent type
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from external.agent.deep_research.deep_researcher import deep_researcher
from external.agent.deep_research.configuration import Configuration, SearchAPI

logger = logging.getLogger(__name__)


def handle_deep_research_query(
    user_query: str,
    conversation_history: list,
    prior_state: Optional[Dict] = None,
    tools_registry: Optional[Any] = None,
    policy_limits: Optional[dict] = None,
    show_thinking: bool = False,
    agent_id: Optional[str] = None,
    on_decider_output: Optional[Callable] = None,
    continuity_packet: Optional[dict] = None,
) -> dict:
    """
    Handle deep research query using Open Deep Research.
    
    Args:
        user_query: User's query string
        conversation_history: List of previous conversation turns
        prior_state: Optional prior state (not used for deep research)
        tools_registry: Tools registry (not used, deep research has its own tools)
        policy_limits: Optional policy limits override
        show_thinking: Whether to show thinking trace
        agent_id: Agent instance ID (for loading config)
        on_decider_output: Optional callback (not used for deep research)
        continuity_packet: Optional continuity packet (not used for deep research)
        
    Returns:
        Response dictionary with status and final_report
    """
    try:
        # Load agent config if agent_id provided
        deep_research_config = {}
        if agent_id:
            try:
                from external.core.db.session import get_db_session
                from external.agent.persistence import get_agent_db
                with get_db_session() as db:
                    agent = get_agent_db(db, agent_id)
                    if agent:
                        deep_research_config = agent.get("deep_research_config") or {}
            except Exception as e:
                logger.warning(f"Error loading agent config for {agent_id}: {e}")
        
        # Build RunnableConfig from agent config + defaults
        configurable = {}
        
        # Model configuration (from agent config or defaults)
        configurable["research_model"] = deep_research_config.get(
            "research_model", 
            "anthropic:claude-3-5-haiku-20241022"  # Default to Haiku for cost savings
        )
        configurable["summarization_model"] = deep_research_config.get(
            "summarization_model",
            "anthropic:claude-3-5-haiku-20241022"
        )
        configurable["compression_model"] = deep_research_config.get(
            "compression_model",
            "anthropic:claude-3-5-haiku-20241022"
        )
        configurable["final_report_model"] = deep_research_config.get(
            "final_report_model",
            "anthropic:claude-3-5-haiku-20241022"
        )
        
        # Model token limits (Haiku max is 8192)
        configurable["research_model_max_tokens"] = deep_research_config.get(
            "research_model_max_tokens", 8192
        )
        configurable["summarization_model_max_tokens"] = deep_research_config.get(
            "summarization_model_max_tokens", 4096
        )
        configurable["compression_model_max_tokens"] = deep_research_config.get(
            "compression_model_max_tokens", 4096
        )
        configurable["final_report_model_max_tokens"] = deep_research_config.get(
            "final_report_model_max_tokens", 8192
        )
        
        # Search API configuration
        search_api_str = deep_research_config.get("search_api", "tavily")
        try:
            configurable["search_api"] = SearchAPI(search_api_str)
        except ValueError:
            logger.warning(f"Invalid search_api '{search_api_str}', defaulting to tavily")
            configurable["search_api"] = SearchAPI.TAVILY
        
        # Research settings
        configurable["max_researcher_iterations"] = deep_research_config.get(
            "max_researcher_iterations", 
            policy_limits.get("max_iterations", 3) if policy_limits else 3
        )
        configurable["max_concurrent_research_units"] = deep_research_config.get(
            "max_concurrent_research_units", 3
        )
        configurable["max_react_tool_calls"] = deep_research_config.get(
            "max_react_tool_calls", 10
        )
        configurable["allow_clarification"] = deep_research_config.get(
            "allow_clarification", False
        )
        configurable["max_structured_output_retries"] = deep_research_config.get(
            "max_structured_output_retries", 3
        )
        configurable["max_content_length"] = deep_research_config.get(
            "max_content_length", 50000
        )
        
        # MCP config (optional)
        if deep_research_config.get("mcp_config"):
            configurable["mcp_config"] = deep_research_config["mcp_config"]
        if deep_research_config.get("mcp_prompt"):
            configurable["mcp_prompt"] = deep_research_config["mcp_prompt"]
        
        # Build input messages from conversation history + current query
        messages = []
        if conversation_history:
            for turn in conversation_history[-5:]:  # Limit to last 5 turns
                if isinstance(turn, dict):
                    if turn.get("query"):
                        messages.append(HumanMessage(content=turn["query"]))
                    # Skip agent responses in history (deep research will generate new)
        
        # Add current query
        messages.append(HumanMessage(content=user_query))
        
        # Create RunnableConfig
        config = RunnableConfig(configurable=configurable)
        
        # Execute deep research graph
        logger.info(f"Starting deep research query: '{user_query[:50]}...'")
        result = asyncio.run(
            deep_researcher.ainvoke(
                {"messages": messages},
                config=config
            )
        )
        
        # Extract final report
        final_report = result.get("final_report", "")
        if not final_report:
            # Fallback: get from messages
            messages_result = result.get("messages", [])
            if messages_result:
                final_report = messages_result[-1].content if hasattr(messages_result[-1], 'content') else str(messages_result[-1])
        
        if not final_report:
            return {
                "status": "ERROR",
                "reason": "No final report generated",
                "error_type": "NO_REPORT"
            }
        
        logger.info(f"Deep research completed, report length: {len(final_report)}")
        
        return {
            "status": "SUCCESS",
            "final_report": final_report,
            "result_summary": final_report,  # For compatibility
            "messages": result.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Deep research query failed: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "reason": str(e),
            "error_type": "DEEP_RESEARCH_ERROR"
        }

