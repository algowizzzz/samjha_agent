"""
Quick Search Agent - Direct Tavily Search for Financial News
Simple, fast search focused on entities (sectors, counterparties, regulatory bodies, etc.)
"""

import logging
from typing import Dict, Any, Optional, List

from tools.tools_registry import ToolsRegistry
from tools.impl.tavily_tool_refactored import TavilyWebSearchTool

logger = logging.getLogger(__name__)

# Financial News Domains - Primary Sources
FINANCIAL_NEWS_DOMAINS = [
    # Major Financial News Outlets
    "bloomberg.com",
    "reuters.com",
    "wsj.com",  # Wall Street Journal
    "ft.com",  # Financial Times
    "cnbc.com",
    "marketwatch.com",
    "yahoo.com/finance",
    "finance.yahoo.com",
    "investing.com",
    "seekingalpha.com",
    "barrons.com",
    "forbes.com",
    "businessinsider.com",
    
    # Regulatory Bodies & Official Sources
    "sec.gov",  # US Securities and Exchange Commission
    "federalreserve.gov",
    "fdic.gov",  # Federal Deposit Insurance Corporation
    "occ.gov",  # Office of the Comptroller of the Currency
    "osfi-bsif.gc.ca",  # Office of the Superintendent of Financial Institutions (Canada)
    "bankofengland.co.uk",
    "ecb.europa.eu",  # European Central Bank
    "bis.org",  # Bank for International Settlements
    "fsb.org",  # Financial Stability Board
    
    # Financial Data & Analytics
    "morningstar.com",
    "fitchratings.com",
    "moodys.com",
    "spglobal.com",  # S&P Global
    "fitchratings.com",
    
    # Industry Publications
    "americanbanker.com",
    "bankingjournal.com",
    "risk.net",
    "thebanker.com",
    
    # Regional Financial News
    "scmp.com",  # South China Morning Post (Asia focus)
    "afr.com",  # Australian Financial Review
    "ft.com",  # Financial Times (already listed, but important)
]

# Blocked Domains (Social Media, Non-News)
BLOCKED_DOMAINS = [
    "reddit.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "linkedin.com",  # Sometimes has news but not primary source
    "instagram.com",
    "tiktok.com",
    "youtube.com",  # Video content, not quick news
]


def load_quick_search_config(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load quick search configuration from agent DB.
    
    Returns:
        Configuration dict with max_results, search_depth, include_answer, etc.
    """
    default_config = {
        "max_results": 5,
        "search_depth": "basic",  # Fast search
        "include_answer": True,  # Get AI-generated summary
        "include_domains": FINANCIAL_NEWS_DOMAINS,
        "exclude_domains": BLOCKED_DOMAINS,
    }
    
    if not agent_id:
        return default_config
    
    try:
        from external.core.db.session import get_db_session
        from external.agent.persistence import get_agent_db
        
        with get_db_session() as db:
            agent = get_agent_db(db, agent_id)
            if agent:
                # Load Tavily API key
                tavily_api_key = agent.get("tavily_api_key")
                
                # Load quick_search_config override
                quick_search_config = agent.get("quick_search_config") or {}
                
                # Merge defaults with agent-specific config
                # If agent has include_domains/exclude_domains in config, use those (override defaults)
                final_config = {**default_config}
                if "include_domains" in quick_search_config:
                    final_config["include_domains"] = quick_search_config["include_domains"]
                if "exclude_domains" in quick_search_config:
                    final_config["exclude_domains"] = quick_search_config["exclude_domains"]
                # Merge other config fields
                final_config.update({k: v for k, v in quick_search_config.items() if k not in ("include_domains", "exclude_domains")})
                final_config["tavily_api_key"] = tavily_api_key
                
                return final_config
    except Exception as e:
        logger.warning(f"Error loading quick_search_config for agent {agent_id}: {e}")
    
    return default_config


def format_quick_search_response(
    query: str,
    tavily_result: Dict[str, Any],
    entity_focus: bool = True
) -> str:
    """
    Format Tavily search results into a natural language response.
    
    Args:
        query: Original user query
        tavily_result: Result from TavilyWebSearchTool
        entity_focus: Whether to emphasize entities (sectors, counterparties, etc.)
    
    Returns:
        Formatted response string with citations
    """
    # Extract components
    answer = tavily_result.get("answer", "")
    results = tavily_result.get("results", [])
    results_count = tavily_result.get("results_count", len(results))
    
    # Build response
    response_parts = []
    
    # Use Tavily's AI-generated answer if available
    if answer:
        response_parts.append(answer)
        response_parts.append("")  # Blank line
    
    # Add entity-focused context if applicable
    if entity_focus and results:
        response_parts.append("**Key Information:**")
        response_parts.append("")
    
    # Add top results with citations
    if results:
        response_parts.append("**Sources:**")
        response_parts.append("")
        
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            score = result.get("score", 0)
            
            # Format as numbered list with citation
            citation = f"{i}. [{title}]({url})"
            if score > 0:
                citation += f" (relevance: {score:.2f})"
            
            response_parts.append(citation)
            
            # Add brief excerpt if content available
            if content and len(content) > 0:
                excerpt = content[:200] + "..." if len(content) > 200 else content
                response_parts.append(f"   {excerpt}")
                response_parts.append("")
    
    # Add summary footer
    if results_count > 0:
        response_parts.append("")
        response_parts.append(f"*Found {results_count} relevant result(s) from financial news sources.*")
    
    return "\n".join(response_parts)


def handle_quick_search_query(
    user_query: str,
    conversation_history: list,
    prior_state: Optional[Dict[str, Any]] = None,
    tools_registry: Optional[Any] = None,
    policy_limits: Optional[dict] = None,
    show_thinking: bool = False,
    agent_id: Optional[str] = None,
    on_decider_output: Optional[Any] = None,  # Not used for quick search
    continuity_packet: Optional[dict] = None,  # Not used for quick search
) -> dict:
    """
    Handle a quick search query using direct Tavily search.
    
    This is a simple, fast search agent that:
    1. Takes user query
    2. Searches Tavily with financial news domain restrictions
    3. Formats and returns results immediately
    
    No iterations, no planning, no complex processing.
    """
    logger.info(f"Starting quick search query for agent {agent_id}: {user_query[:100]}...")
    
    try:
        # Load configuration
        config = load_quick_search_config(agent_id)
        tavily_api_key = config.get("tavily_api_key")
        
        if not tavily_api_key:
            logger.error(f"No Tavily API key configured for agent {agent_id}")
            return {
                "status": "ERROR",
                "error_type": "MISSING_API_KEY",
                "last_error": "Tavily API key not configured. Please configure it in the agent settings.",
                "result_summary": "Quick search requires a Tavily API key to be configured.",
            }
        
        # Initialize tools registry if needed
        if tools_registry is None:
            tools_registry = ToolsRegistry()
        
        # Create Tavily search tool with API key
        tavily_tool = TavilyWebSearchTool({
            "api_key": tavily_api_key,
            "name": "tavily_web_search",
            "description": "Quick financial news search",
            "enabled": True,
        })
        
        # Prepare search parameters
        search_params = {
            "query": user_query,
            "search_depth": config.get("search_depth", "basic"),
            "max_results": config.get("max_results", 5),
            "include_answer": config.get("include_answer", True),
            "include_images": False,  # Not needed for quick search
        }
        
        # Add domain filters
        include_domains = config.get("include_domains")
        exclude_domains = config.get("exclude_domains")
        
        # Ensure domains are lists (Tavily expects list format)
        if include_domains:
            if isinstance(include_domains, str):
                # Convert comma/newline-separated string to list
                include_domains = [d.strip() for d in include_domains.replace('\n', ',').split(',') if d.strip()]
            elif not isinstance(include_domains, list):
                include_domains = [str(include_domains)]
            search_params["include_domains"] = include_domains
            logger.info(f"Domain filter: ONLY searching {len(include_domains)} domain(s): {include_domains}")
        
        if exclude_domains:
            if isinstance(exclude_domains, str):
                exclude_domains = [d.strip() for d in exclude_domains.replace('\n', ',').split(',') if d.strip()]
            elif not isinstance(exclude_domains, list):
                exclude_domains = [str(exclude_domains)]
            search_params["exclude_domains"] = exclude_domains
            logger.info(f"Domain filter: Excluding {len(exclude_domains)} domain(s): {exclude_domains}")
        
        # Execute search
        logger.info(f"Executing Tavily search with params: query='{user_query}', include_domains={include_domains if include_domains else 'None'}, exclude_domains={exclude_domains if exclude_domains else 'None'}")
        tavily_result = tavily_tool.execute(search_params)
        
        # Format response
        formatted_response = format_quick_search_response(
            user_query,
            tavily_result,
            entity_focus=True  # Focus on entities (sectors, counterparties, etc.)
        )
        
        # Extract sources for citation
        sources = []
        for result in tavily_result.get("results", []):
            sources.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0),
            })
        
        logger.info(f"Quick search completed, found {len(sources)} sources")
        
        return {
            "status": "SUCCESS",
            "final_answer": formatted_response,
            "result_summary": formatted_response[:500] + "..." if len(formatted_response) > 500 else formatted_response,
            "sources": sources,
            "tavily_result": tavily_result,  # Include raw result for debugging
        }
        
    except Exception as e:
        logger.error(f"Quick search agent failed: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "error_type": "QUICK_SEARCH_ERROR",
            "last_error": str(e),
            "result_summary": f"Quick search failed: {str(e)[:200]}...",
        }

