"""
Shared utility for loading prompts from external/config/prompts/.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from external/config/prompts/{prompt_name}.md
    
    Args:
        prompt_name: Name of the prompt file (without .md extension)
        
    Returns:
        Prompt text as string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_path = Path(f"external/config/prompts/{prompt_name}.md")
    
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        error_msg = f"Prompt file not found: {prompt_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

