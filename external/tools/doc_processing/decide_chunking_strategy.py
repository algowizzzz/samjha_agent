from __future__ import annotations

from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool


DEFAULT_PAGE_THRESHOLD = 10
DEFAULT_STRATEGY = "by_h2"
DEFAULT_CONTEXT_AWARE_LEVEL = "h2"


class DecideChunkingStrategyTool(BaseMCPTool):
    """Determine whether and how to chunk the document."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "decide_chunking_strategy",
            "description": "Decides if chunking is required and which strategy to use based on metadata and config.",
            "inputSchema": self.get_input_schema(),
            "outputSchema": self.get_output_schema(),
        }
        if config:
            tool_config.update(config)
        super().__init__(tool_config)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_metadata": {"type": "object"},
                "config": {
                    "type": "object",
                    "properties": {
                        "page_threshold": {"type": ["integer", "null"]},
                        "context_aware_level": {"type": ["string", "null"]},
                        "default_strategy": {"type": ["string", "null"]},
                        "force_chunk": {"type": ["boolean", "null"]},
                    },
                    "additionalProperties": True,
                },
            },
            "required": ["file_metadata", "config"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "should_chunk": {"type": "boolean"},
                "strategy": {"type": "string"},
                "reason": {"type": "string"},
                "context_aware_level": {"type": "string"},
            },
            "required": ["should_chunk", "strategy", "reason", "context_aware_level"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        metadata = arguments["file_metadata"]
        cfg = arguments.get("config", {}) or {}

        page_threshold = cfg.get("page_threshold") or DEFAULT_PAGE_THRESHOLD
        context_level = cfg.get("context_aware_level") or DEFAULT_CONTEXT_AWARE_LEVEL

        page_count = metadata.get("page_count", 0) or 0

        # Always follow page threshold (no force_chunk override)
        should_chunk = page_count >= page_threshold
        reason = "page_count>=threshold" if should_chunk else "below_threshold"

        # If should_chunk is true, derive strategy from context_aware_level
        if should_chunk:
            if context_level and context_level.lower().startswith("h") and context_level[1:].isdigit():
                strategy = f"by_{context_level.lower()}"
            else:
                # Fallback to default if context_level is invalid
                strategy = DEFAULT_STRATEGY
        else:
            strategy = "none"

        self.logger.info(
            "decide_chunking_strategy: file_id=%s pages=%s should_chunk=%s strategy=%s reason=%s",
            metadata.get("file_id"),
            page_count,
            should_chunk,
            strategy,
            reason,
        )

        return {
            "should_chunk": should_chunk,
            "strategy": strategy,
            "reason": reason,
            "context_aware_level": context_level,
        }
