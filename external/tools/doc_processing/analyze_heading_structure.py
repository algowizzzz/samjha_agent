from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool

from .utils import headings_to_levels, parse_markdown_headings


class AnalyzeHeadingStructureTool(BaseMCPTool):
    """Analyse markdown headings to understand document structure."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "analyze_heading_structure",
            "description": "Parses markdown headings (H1-H6) and returns structural metadata.",
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
                "headings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading_level": {"type": "string"},
                            "heading_text": {"type": "string"},
                            "heading_path": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "line_number": {"type": "integer"},
                        },
                        "required": ["heading_level", "heading_text", "heading_path", "line_number"],
                    },
                },
                "heading_levels": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["file_id", "md_file_id", "headings", "heading_levels"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw_markdown = arguments["raw_markdown"]
        parsed_headings: List[Dict[str, Any]] = parse_markdown_headings(raw_markdown)
        heading_levels = headings_to_levels(parsed_headings)

        self.logger.info(
            "analyze_heading_structure: file_id=%s headings=%d levels=%s",
            arguments["file_id"],
            len(parsed_headings),
            ",".join(heading_levels),
        )

        return {
            "file_id": arguments["file_id"],
            "md_file_id": arguments["md_file_id"],
            "headings": parsed_headings,
            "heading_levels": heading_levels,
        }
