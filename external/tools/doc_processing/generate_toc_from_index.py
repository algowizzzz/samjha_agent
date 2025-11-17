from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


def _level_to_int(level: str) -> int:
    if level.lower().startswith("h") and level[1:].isdigit():
        return int(level[1:])
    return 6


class GenerateTocFromIndexTool(BaseMCPTool):
    """Generate a markdown table of contents from the index structure."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "generate_toc_from_index",
            "description": "Generates a markdown TOC using the ordered sections in the index.",
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
                "index": {"type": "object"},
                "config": {
                    "type": "object",
                    "properties": {
                        "toc_level": {"type": ["string", "null"]},
                    },
                    "additionalProperties": True,
                },
            },
            "required": ["index"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "toc_markdown": {"type": "string"},
                "toc": {"type": "array"},
            },
            "required": ["toc_markdown", "toc"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        index = arguments["index"]
        cfg = arguments.get("config", {}) or {}
        toc_level = cfg.get("toc_level") or "h2"
        max_level = _level_to_int(toc_level)

        toc_entries: List[str] = ["## Table of Contents"]
        toc_metadata: List[Dict[str, Any]] = []

        for section in sorted(index.get("sections", []), key=lambda s: s.get("order", 0)):
            level = _level_to_int(section.get("heading_level", "h2"))
            if level > max_level:
                continue
            indent = "  " * (level - 1)
            anchor = section.get("anchor", "")
            heading = section.get("heading_text_improved") or section.get("heading_text_original") or "Section"
            toc_entries.append(f"{indent}- [{heading}](#{anchor})")
            toc_metadata.append(
                {
                    "file_section_id": section.get("file_section_id"),
                    "heading": heading,
                    "order": section.get("order", 0),
                    "anchor": anchor,
                }
            )

        toc_markdown = "\n".join(toc_entries) if len(toc_entries) > 1 else ""

        self.logger.info(
            "generate_toc_from_index: file_id=%s entries=%d",
            index.get("file_id"),
            len(toc_metadata),
        )

        return {
            "toc_markdown": toc_markdown,
            "toc": toc_metadata,
        }
