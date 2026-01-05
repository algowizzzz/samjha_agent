"""
Web Research Synthesis - Generate final answer from EvidencePack.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from external.platform.llm import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def load_web_research_synthesis_prompt(agent_id: Optional[str] = None) -> str:
    """Load Web Research Synthesis prompt from DB or file."""
    if agent_id:
        try:
            from external.core.db.session import get_db_session
            from external.agent.persistence import get_prompt_content
            with get_db_session() as db:
                prompt_content = get_prompt_content(db, "web_research_synthesis", category="web_search")
                if prompt_content:
                    return prompt_content
        except Exception as e:
            logger.warning(f"Failed to load synthesis prompt from DB for agent {agent_id}: {e}")
    
    # Fallback to file
    from pathlib import Path
    prompt_path = Path("external/config/prompts/web_research_synthesis.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        logger.warning(f"Web Research Synthesis prompt not found at {prompt_path}")
        return "Synthesize the research findings into a comprehensive answer."


def synthesize_final_answer(
    user_query: str,
    research_spec: dict,
    evidence_pack: dict,
    conversation_history: list,
    agent_id: Optional[str] = None,
    agent_model: Optional[str] = None
) -> str:
    """
    Synthesize final answer from EvidencePack using LLM.
    
    Args:
        user_query: Original user query
        research_spec: ResearchSpec from Decider
        evidence_pack: EvidencePack from Executor
        conversation_history: Conversation history
        agent_id: Agent instance ID
        agent_model: Agent's configured model
        
    Returns:
        Final synthesized answer string
    """
    if not LLM_AVAILABLE:
        # Fallback: simple summary
        sources = evidence_pack.get("sources", [])
        claims = evidence_pack.get("claims", [])
        return f"Research completed. Found {len(sources)} sources with {len(claims)} claims."
    
    llm_client = get_llm_client()
    if not llm_client.is_available():
        return f"Research completed. Found {len(evidence_pack.get('sources', []))} sources."
    
    prompt_template = load_web_research_synthesis_prompt(agent_id)
    
    import json
    context = {
        "user_query": user_query,
        "user_question": research_spec.get("user_question", user_query),
        "research_spec": json.dumps(research_spec, indent=2),
        "evidence_pack": json.dumps(evidence_pack, indent=2),
        "conversation_history": json.dumps(conversation_history, indent=2),
    }
    
    prompt = f"""{prompt_template}

## INPUTS

User Query: {context['user_query']}
Research Question: {context['user_question']}

Research Spec:
{context['research_spec']}

Evidence Pack:
{context['evidence_pack']}

Conversation History:
{context['conversation_history']}

---

Synthesize the final answer based on the EvidencePack above.
"""
    
    try:
        model = agent_model or "claude-3-sonnet-20240229"
        response = llm_client.invoke_with_prompt(
            system_prompt="",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=4000,
            response_format="text",
            model=model
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        # Fallback
        sources = evidence_pack.get("sources", [])
        claims = evidence_pack.get("claims", [])
        return f"Research completed. Found {len(sources)} sources with {len(claims)} claims. Synthesis failed: {str(e)}"

