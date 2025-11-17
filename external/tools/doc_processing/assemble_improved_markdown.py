from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool

from .utils import generate_md_file_id, write_text_file

_IMPROVED_OUTPUT_DIR = Path("external/data/doc_review/improved")


class AssembleImprovedMarkdownTool(BaseMCPTool):
    """Assemble the improved markdown document using chunk data and index metadata."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "assemble_improved_markdown",
            "description": "Assembles the improved markdown document, applying title updates and ordering sections.",
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
                "chunks": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "index": {"type": "object"},
                "toc_markdown": {"type": ["string", "null"]},
            },
            "required": ["file_id", "chunks", "index"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "improved_md_file_id": {"type": "string"},
                "markdown": {"type": "string"},
                "improved_md_path": {"type": "string"},
            },
            "required": ["improved_md_file_id", "markdown", "improved_md_path"],
            "additionalProperties": False,
        }

    def _heading_line(self, level: str, title: str) -> str:
        try:
            level_int = int(level.lower().replace("h", ""))
        except Exception:
            level_int = 2
        level_int = max(1, min(level_int, 6))
        return f"{'#' * level_int} {title.strip()}"

    def _strip_existing_heading(self, text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("#"):
            return "\n".join(lines[1:]).strip()
        return text.strip()

    def _ordered_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        primary_status = {"mapped", "appended_late"}
        primary = [s for s in sections if s.get("status") in primary_status]
        secondary = [s for s in sections if s.get("status") not in primary_status]
        primary.sort(key=lambda s: s.get("order", 0))
        secondary.sort(key=lambda s: s.get("order", 0))
        return primary + secondary

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_id = arguments["file_id"]
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in arguments["chunks"]}
        sections = self._ordered_sections(arguments["index"].get("sections", []))

        toc_markdown = arguments.get("toc_markdown") or ""
        assembled: List[str] = []
        if toc_markdown:
            assembled.append(toc_markdown.strip())
            assembled.append("")

        for section in sections:
            chunk = chunks_by_id.get(section["chunk_id"])
            if not chunk:
                continue
            heading_level = section.get("heading_level", chunk.get("heading_level", "h2"))
            heading_text = (
                section.get("heading_text_improved")
                or section.get("heading_text_original")
                or chunk.get("heading_text_original")
                or "Section"
            )
            body = self._strip_existing_heading(chunk.get("text", ""))
            heading_line = self._heading_line(heading_level, heading_text)
            assembled.append(heading_line)
            if body:
                assembled.append(body)
            assembled.append("")

        markdown = "\n".join(assembled).strip() + "\n"
        improved_md_file_id = generate_md_file_id(f"{file_id}_improved")
        improved_path = _IMPROVED_OUTPUT_DIR / f"{improved_md_file_id}.md"
        write_text_file(improved_path, markdown)

        self.logger.info(
            "assemble_improved_markdown: file_id=%s sections=%d path=%s",
            file_id,
            len(sections),
            improved_path,
        )

        return {
            "improved_md_file_id": improved_md_file_id,
            "markdown": markdown,
            "improved_md_path": str(improved_path),
        }
