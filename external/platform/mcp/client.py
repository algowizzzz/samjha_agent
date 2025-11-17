"""
Platform MCP Client - Unified MCP tool execution.

Provides a single interface for executing MCP tools across all products.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tools.tools_registry import ToolsRegistry

logger = logging.getLogger(__name__)


class MCPClient:
    """Unified MCP client for executing tools across all products."""

    def __init__(self, tools_registry: Optional[ToolsRegistry] = None):
        """
        Initialize MCP client.
        
        Args:
            tools_registry: Optional tools registry. If None, creates a new one.
        """
        self.registry = tools_registry or ToolsRegistry()

    def call(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool.
        
        Args:
            tool_name: Name of the tool to execute
            payload: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        logger.debug("Calling MCP tool %s with payload keys=%s", tool_name, list(payload.keys()))
        return tool.execute_with_tracking(payload)


# Global default client
_default_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get the default MCP client instance."""
    global _default_client
    if _default_client is None:
        _default_client = MCPClient()
    return _default_client


def call_mcp(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to call an MCP tool using the default client.
    
    Args:
        tool_name: Name of the tool to execute
        payload: Tool arguments
        
    Returns:
        Tool execution result
    """
    return get_mcp_client().call(tool_name, payload)

