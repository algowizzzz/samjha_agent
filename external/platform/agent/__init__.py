"""
Platform agent module - Base agent framework.

Provides base classes and utilities for building agents across all products.
"""

from external.platform.agent.base_agent import LangGraphAgentTool
from external.platform.agent.streaming import StreamingAgent

__all__ = [
    "LangGraphAgentTool",
    "StreamingAgent",
]

