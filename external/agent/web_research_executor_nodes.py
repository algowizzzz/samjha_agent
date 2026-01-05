"""
Web Research Executor Nodes - Individual nodes for the executor graph.
"""

import logging
import json
from typing import Dict, Any, Optional
from external.agent.state_types import ResearchExecutorState

logger = logging.getLogger(__name__)


def _get_tavily_tool_with_api_key(tools_registry, tool_name: str, api_key: Optional[str] = None):
    """
    Get Tavily tool instance, creating a new one with agent's API key if provided.
    """
    tool = tools_registry.get_tool(tool_name)
    if not tool:
        return None
    
    # If API key is provided and tool is a Tavily tool, create a new instance with the API key
    if api_key and tool_name.startswith("tavily_"):
        try:
            from tools.impl.tavily_tool_refactored import (
                TavilyWebSearchTool, TavilyNewsSearchTool,
                TavilyResearchSearchTool, TavilyDomainSearchTool
            )
            
            # Get tool config from registry
            tool_config = tools_registry.tool_configs.get(tool_name, {})
            # Override API key
            tool_config = tool_config.copy()
            tool_config['api_key'] = api_key
            
            # Create new instance with agent's API key
            tool_class_map = {
                "tavily_web_search": TavilyWebSearchTool,
                "tavily_news_search": TavilyNewsSearchTool,
                "tavily_research_search": TavilyResearchSearchTool,
                "tavily_domain_search": TavilyDomainSearchTool,
            }
            
            tool_class = tool_class_map.get(tool_name)
            if tool_class:
                return tool_class(tool_config)
        except Exception as e:
            logger.warning(f"Failed to create tool instance with API key: {e}")
            # Fallback to original tool
            return tool
    
    return tool


def research_discovery_node(state: ResearchExecutorState, tools_registry) -> Dict[str, Any]:
    """
    Execute Tavily tools from research_plan and collect sources.
    """
    research_plan = state.get("research_plan", [])
    sources_seen = state.get("sources_seen", [])
    tavily_api_key = state.get("tavily_api_key")
    
    sources = []
    
    for step in research_plan:
        tool_name = step.get("tool")
        args = step.get("args", {})
        
        try:
            # Get tool from registry (with API key if provided)
            tool = _get_tavily_tool_with_api_key(tools_registry, tool_name, tavily_api_key)
            if not tool:
                logger.warning(f"Tool {tool_name} not found in registry")
                continue
            
            # Execute tool
            result = tool.execute(args)
            
            # Extract sources from result
            # Tavily tools return: {"query": "...", "results_count": N, "results": [...], "answer": "..."}
            if isinstance(result, dict):
                if "results" in result:
                    # Tavily format: {"results": [{"url": "...", "title": "...", "content": "...", "score": 0.9, "published_date": "..."}]}
                    for item in result.get("results", []):
                        url = item.get("url")
                        if url and url not in sources_seen:
                            # Extract content (prefer content over raw_content for snippet)
                            content = item.get("content", "") or item.get("raw_content", "")
                            sources.append({
                                "url": url,
                                "title": item.get("title", ""),
                                "snippet": content[:500] if content else "",  # First 500 chars
                                "published_date": item.get("published_date"),
                                "authority_score": item.get("score", 0.5),  # Use Tavily's score as initial authority
                                "source_type": "general"
                            })
                            sources_seen.append(url)
                elif "url" in result:
                    # Single result (fallback format)
                    url = result.get("url")
                    if url and url not in sources_seen:
                        content = result.get("content", "") or result.get("raw_content", "")
                        sources.append({
                            "url": url,
                            "title": result.get("title", ""),
                            "snippet": content[:500] if content else "",
                            "published_date": result.get("published_date"),
                            "authority_score": result.get("score", 0.5),
                            "source_type": "general"
                        })
                        sources_seen.append(url)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            continue
    
    # Update evidence_pack with sources
    evidence_pack = state.get("evidence_pack", {"sources": [], "claims": [], "conflicts": [], "gaps": []})
    evidence_pack["sources"].extend(sources)
    
    return {
        "evidence_pack": evidence_pack,
        "sources_seen": sources_seen,
    }


def claim_extraction_node(state: ResearchExecutorState, tools_registry, domain_md: str) -> Dict[str, Any]:
    """
    Extract claims from sources using LLM.
    """
    evidence_pack = state.get("evidence_pack", {"sources": [], "claims": [], "conflicts": [], "gaps": []})
    sources = evidence_pack.get("sources", [])
    research_spec = state.get("research_spec", {})
    
    claims = []
    
    # Use LLM to extract claims (simplified - batch process sources)
    try:
        from external.platform.llm import get_llm_client
        llm_client = get_llm_client()
        
        if llm_client.is_available():
            # Load claim extraction prompt from DB or file
            prompt_template = None
            agent_id = state.get("agent_id")  # Get from research_spec if available
            if agent_id:
                try:
                    from external.core.db.session import get_db_session
                    from external.agent.persistence import get_prompt_content
                    with get_db_session() as db:
                        prompt_template = get_prompt_content(db, "web_research_claim_extraction", category="web_search")
                except Exception as e:
                    logger.warning(f"Failed to load claim extraction prompt from DB: {e}")
            
            if not prompt_template:
                # Fallback to file
                from pathlib import Path
                prompt_path = Path("external/config/prompts/web_research_claim_extraction.md")
                if prompt_path.exists():
                    prompt_template = prompt_path.read_text()
                else:
                    prompt_template = "Extract factual claims from the sources."
            
            # Batch process sources (simplified: process first 10 sources)
            sources_to_process = sources[:10]
            
            for source in sources_to_process:
                prompt = f"""{prompt_template}

Source:
URL: {source.get('url')}
Title: {source.get('title')}
Content: {source.get('snippet')}

Research Spec:
{json.dumps(research_spec, indent=2)}

Extract claims as JSON:
"""
                try:
                    response = llm_client.invoke_with_prompt(
                        system_prompt="",
                        user_prompt=prompt,
                        temperature=0.0,
                        max_tokens=2000,
                        response_format="json",
                        model="claude-3-haiku-20240307"  # Use Haiku for extraction (faster, cheaper)
                    )
                    result = json.loads(response)
                    if "claims" in result:
                        for claim in result["claims"]:
                            claim["extracted_from"] = source.get("url")
                            claims.append(claim)
                except Exception as e:
                    logger.warning(f"Failed to extract claims from {source.get('url')}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}", exc_info=True)
    
    # Update evidence_pack with claims
    evidence_pack["claims"].extend(claims)
    
    return {
        "evidence_pack": evidence_pack,
    }


def conflict_detection_node(state: ResearchExecutorState, tools_registry, domain_md: str) -> Dict[str, Any]:
    """
    Detect conflicts between claims using LLM.
    """
    evidence_pack = state.get("evidence_pack", {"sources": [], "claims": [], "conflicts": [], "gaps": []})
    claims = evidence_pack.get("claims", [])
    
    conflicts = []
    
    # Use LLM to detect conflicts (simplified - compare claim pairs)
    try:
        from external.platform.llm import get_llm_client
        llm_client = get_llm_client()
        
        if llm_client.is_available() and len(claims) > 1:
            # Load conflict detection prompt from DB or file
            prompt_template = None
            agent_id = state.get("agent_id")  # Get from research_spec if available
            if agent_id:
                try:
                    from external.core.db.session import get_db_session
                    from external.agent.persistence import get_prompt_content
                    with get_db_session() as db:
                        prompt_template = get_prompt_content(db, "web_research_conflict_detection", category="web_search")
                except Exception as e:
                    logger.warning(f"Failed to load conflict detection prompt from DB: {e}")
            
            if not prompt_template:
                # Fallback to file
                from pathlib import Path
                prompt_path = Path("external/config/prompts/web_research_conflict_detection.md")
                if prompt_path.exists():
                    prompt_template = prompt_path.read_text()
                else:
                    prompt_template = "Detect conflicts between claims."
            
            # Compare claim pairs (simplified: compare first 10 claims)
            claims_to_check = claims[:10]
            
            for i, claim1 in enumerate(claims_to_check):
                for claim2 in claims_to_check[i+1:]:
                    prompt = f"""{prompt_template}

Claims:
{json.dumps({"claims": [claim1, claim2]}, indent=2)}

Detect conflicts as JSON:
"""
                    try:
                        response = llm_client.invoke_with_prompt(
                            system_prompt="",
                            user_prompt=prompt,
                            temperature=0.0,
                            max_tokens=1000,
                            response_format="json",
                            model="claude-3-haiku-20240307"
                        )
                        result = json.loads(response)
                        if "conflicts" in result:
                            conflicts.extend(result["conflicts"])
                    except Exception as e:
                        logger.warning(f"Failed to detect conflicts: {e}")
                        continue
    except Exception as e:
        logger.error(f"Conflict detection failed: {e}", exc_info=True)
    
    # Update evidence_pack with conflicts
    evidence_pack["conflicts"].extend(conflicts)
    
    return {
        "evidence_pack": evidence_pack,
    }


def evidence_synthesis_node(state: ResearchExecutorState, tools_registry, domain_md: str) -> Dict[str, Any]:
    """
    Build final EvidencePack structure.
    """
    evidence_pack = state.get("evidence_pack", {"sources": [], "claims": [], "conflicts": [], "gaps": []})
    
    # Score sources (simplified)
    for source in evidence_pack.get("sources", []):
        url = source.get("url", "")
        # Simple authority scoring
        if ".gov" in url or ".edu" in url:
            source["authority_score"] = 0.9
            source["source_type"] = "official" if ".gov" in url else "academic"
        elif "arxiv.org" in url or "nature.com" in url:
            source["authority_score"] = 0.8
            source["source_type"] = "academic"
        else:
            source["authority_score"] = 0.5
            source["source_type"] = "general"
    
    # Identify gaps (simplified - check if research_spec requirements are met)
    research_spec = state.get("research_spec", {})
    quality_bar = research_spec.get("quality_bar", {})
    min_sources = quality_bar.get("min_sources", 0)
    
    gaps = []
    if len(evidence_pack.get("sources", [])) < min_sources:
        gaps.append({
            "gap_description": f"Only found {len(evidence_pack.get('sources', []))} sources, need {min_sources}",
            "criticality": "high" if min_sources > 5 else "medium"
        })
    
    evidence_pack["gaps"] = gaps
    
    return {
        "evidence_pack": evidence_pack,
    }


def outcome_node(state: ResearchExecutorState) -> Dict[str, Any]:
    """
    Generate final executor report.
    """
    evidence_pack = state.get("evidence_pack", {"sources": [], "claims": [], "conflicts": [], "gaps": []})
    
    # Determine status
    status = "SUCCESS"
    error_type = None
    
    if state.get("halt_execution"):
        status = "ERROR"
        error_type = "HALTED"
    elif state.get("last_error"):
        status = "ERROR"
        error_type = "EXECUTION_ERROR"
    
    executor_report = {
        "status": status,
        "error_type": error_type,
        "evidence_pack": evidence_pack,
        "sources_count": len(evidence_pack.get("sources", [])),
        "claims_count": len(evidence_pack.get("claims", [])),
        "conflicts_count": len(evidence_pack.get("conflicts", [])),
        "gaps_count": len(evidence_pack.get("gaps", [])),
        "last_error": state.get("last_error"),
    }
    
    return {
        "executor_report": executor_report,
    }

