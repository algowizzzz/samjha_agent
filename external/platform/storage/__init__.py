"""
Platform storage module - State and session management.

Provides unified storage abstractions for all products.
"""

from external.platform.storage.state_manager import AgentStateManager
from external.platform.storage.session_manager import AgentSessionManager

__all__ = [
    "AgentStateManager",
    "AgentSessionManager",
]

