"""
Shared utility for loading prompts from DB (with agent overrides) or files.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str, agent_id: Optional[str] = None, category: str = "structured") -> str:
    """
    Load a prompt with priority: agent override > global DB > file.
    
    Args:
        prompt_name: Name of the prompt (without .md extension)
        agent_id: Optional agent ID for agent-specific overrides
        category: Prompt category ("structured" or "web_search")
        
    Returns:
        Prompt text as string
        
    Raises:
        FileNotFoundError: If prompt not found anywhere
    """
    # 1. Try DB first (checks agent override, then global)
    try:
        from external.core.db.session import get_db_session
        from external.agent.persistence import get_prompt_content
        with get_db_session() as db:
            prompt_content = get_prompt_content(db, prompt_name, category=category, agent_id=agent_id)
            if prompt_content:
                logger.debug(f"Loaded prompt '{prompt_name}' from DB (agent_id={agent_id})")
                return prompt_content
    except Exception as e:
        logger.warning(f"Failed to load prompt '{prompt_name}' from DB: {e}")
    
    # 2. Fallback to file
    prompt_path = Path(f"external/config/prompts/{prompt_name}.md")
    if prompt_path.exists():
        logger.debug(f"Loaded prompt '{prompt_name}' from file")
        return prompt_path.read_text()
    
    error_msg = f"Prompt '{prompt_name}' not found in DB or file ({prompt_path})"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

