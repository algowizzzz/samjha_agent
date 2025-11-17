from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tools.tools_registry import ToolsRegistry

logger = logging.getLogger(__name__)


class DocReviewMCPClient:
    """Lightweight wrapper for executing document-review MCP tools."""

    def __init__(self, tools_registry: Optional[ToolsRegistry] = None):
        self.registry = tools_registry or ToolsRegistry()

    def call(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        logger.debug("Calling MCP tool %s with payload keys=%s", tool_name, list(payload.keys()))
        return tool.execute_with_tracking(payload)


_default_client: Optional[DocReviewMCPClient] = None


def get_mcp_client() -> DocReviewMCPClient:
    global _default_client
    if _default_client is None:
        _default_client = DocReviewMCPClient()
    return _default_client


def call_mcp(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return get_mcp_client().call(tool_name, payload)
