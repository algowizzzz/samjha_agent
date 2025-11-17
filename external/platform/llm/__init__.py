"""
Platform LLM module - Unified LLM client and utilities.

Provides a single source of truth for all LLM interactions across products.
"""

from external.platform.llm.client import LLMClient, get_llm_client, is_llm_available

__all__ = [
    "LLMClient",
    "get_llm_client",
    "is_llm_available",
]

