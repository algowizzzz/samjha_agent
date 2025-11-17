from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


class AnnotateMarkdownForUITool(BaseMCPTool):
    """Produce UI annotations describing provenance for sections in the improved markdown."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "annotate_markdown_for_ui",
            "description": "Annotates the improved markdown with provenance metadata for UI rendering.",
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
                "markdown": {"type": "string"},
                "index": {"type": "object"},
            },
            "required": ["file_id", "markdown", "index"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "markdown": {"type": "string"},
                "annotations": {"type": "array"},
            },
            "required": ["markdown", "annotations"],
            "additionalProperties": False,
        }

    def _heading_line(self, level: str, title: str) -> str:
        try:
            level_int = int(level.lower().replace("h", ""))
        except Exception:
            level_int = 2
        level_int = max(1, min(level_int, 6))
        return f"{'#' * level_int} {title.strip()}"

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        markdown = arguments["markdown"]
        index = arguments["index"]
        annotations: List[Dict[str, Any]] = []

        for section in index.get("sections", []):
            heading_level = section.get("heading_level", "h2")
            improved = section.get("heading_text_improved")
            original = section.get("heading_text_original")
            title = improved or original or "Section"
            heading_line = self._heading_line(heading_level, title)

            offset = markdown.find(heading_line)
            if offset == -1:
                continue

            provenance = "ai_title" if improved and improved != original else "original"
            span_id = str(uuid.uuid4())
            annotations.append(
                {
                    "span_id": span_id,
                    "chunk_id": section.get("chunk_id"),
                    "offset_start": offset,
                    "offset_end": offset + len(heading_line),
                    "span_type": "heading",
                    "provenance": provenance,
                    "comment_text": None,
                }
            )

        self.logger.info(
            "annotate_markdown_for_ui: file_id=%s annotations=%d",
            arguments["file_id"],
            len(annotations),
        )

        return {
            "markdown": markdown,
            "annotations": annotations,
        }
