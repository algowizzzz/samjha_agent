from __future__ import annotations

from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool

from .utils import estimate_page_count


class ComputeFileStatsTool(BaseMCPTool):
    """Compute basic statistics for a markdown document."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "compute_file_stats",
            "description": "Computes page count and word count for the markdown representation of a document.",
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
                "file_id": {"type": "string"},
                "md_file_id": {"type": "string"},
                "raw_markdown": {"type": "string"},
            },
            "required": ["file_id", "md_file_id", "raw_markdown"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "md_file_id": {"type": "string"},
                "word_count": {"type": "integer"},
                "char_count": {"type": "integer"},
                "page_count": {"type": "integer"},
            },
            "required": ["file_id", "md_file_id", "word_count", "page_count"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw_markdown = arguments["raw_markdown"]
        words = [token for token in raw_markdown.split() if token.strip()]
        word_count = len(words)
        char_count = len(raw_markdown)
        page_count = estimate_page_count(word_count)

        self.logger.info(
            "compute_file_stats: file_id=%s md_file_id=%s words=%d pages=%d",
            arguments["file_id"],
            arguments["md_file_id"],
            word_count,
            page_count,
        )

        return {
            "file_id": arguments["file_id"],
            "md_file_id": arguments["md_file_id"],
            "word_count": word_count,
            "char_count": char_count,
            "page_count": page_count,
        }
