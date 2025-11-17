"""
Platform MCP module - Model Context Protocol client.

Provides unified interface for tool execution across all products.
"""

from external.platform.mcp.client import MCPClient, get_mcp_client, call_mcp

__all__ = [
    "MCPClient",
    "get_mcp_client",
    "call_mcp",
]

