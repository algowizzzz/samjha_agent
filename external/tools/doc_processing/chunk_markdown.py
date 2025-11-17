from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool

from .utils import generate_chunk_id, parse_markdown_headings


class ChunkMarkdownTool(BaseMCPTool):
    """Split markdown into ordered chunks based on the configured strategy."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "chunk_markdown",
            "description": "Splits markdown into deterministic chunks using heading boundaries.",
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
                "strategy": {"type": "string"},
                "context_aware_level": {"type": "string"},
            },
            "required": ["file_id", "md_file_id", "raw_markdown", "strategy", "context_aware_level"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "items": {"type": "object"},
                },
            },
            "required": ["chunks"],
            "additionalProperties": False,
        }

    def _determine_level(self, strategy: str, context_level: str) -> Optional[int]:
        if strategy.startswith("by_h") and strategy[4:].isdigit():
            return int(strategy[4:])
        if context_level.lower().startswith("h") and context_level[1:].isdigit():
            return int(context_level[1:])
        return None

    def _build_chunks(
        self,
        file_id: str,
        raw_markdown: str,
        target_level: Optional[int],
    ) -> List[Dict[str, Any]]:
        lines = raw_markdown.splitlines()
        headings = parse_markdown_headings(raw_markdown)

        if not target_level or target_level < 1:
            target_level = 1

        level_label = f"h{target_level}"
        filtered = [h for h in headings if h["heading_level"] == level_label]

        if not filtered:
            # fallback: single chunk covering entire document
            return [
                {
                    "chunk_id": generate_chunk_id(file_id, 0),
                    "order": 0,
                    "text": raw_markdown,
                    "heading_text_original": headings[0]["heading_text"] if headings else "Document",
                    "heading_level": headings[0]["heading_level"] if headings else "h1",
                    "heading_path": headings[0]["heading_path"] if headings else ["Document"],
                    "start_line": 1,
                    "end_line": len(lines),
                }
            ]

        chunks: List[Dict[str, Any]] = []
        for idx, heading in enumerate(filtered):
            start_line = heading["line_number"] - 1  # zero-based for slicing
            end_line = len(lines)
            if idx + 1 < len(filtered):
                end_line = filtered[idx + 1]["line_number"] - 1

            chunk_text = "\n".join(lines[start_line:end_line]).strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": generate_chunk_id(file_id, len(chunks)),
                    "order": len(chunks),
                    "text": chunk_text,
                    "heading_text_original": heading["heading_text"],
                    "heading_level": heading["heading_level"],
                    "heading_path": heading["heading_path"],
                    "start_line": start_line + 1,
                    "end_line": end_line,
                }
            )

        if not chunks:
            # As a safety net, return the entire doc as one chunk
            return [
                {
                    "chunk_id": generate_chunk_id(file_id, 0),
                    "order": 0,
                    "text": raw_markdown,
                    "heading_text_original": headings[0]["heading_text"] if headings else "Document",
                    "heading_level": headings[0]["heading_level"] if headings else "h1",
                    "heading_path": headings[0]["heading_path"] if headings else ["Document"],
                    "start_line": 1,
                    "end_line": len(lines),
                }
            ]

        return chunks

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        strategy = arguments["strategy"].lower()
        context_level = arguments["context_aware_level"].lower()
        target_level = self._determine_level(strategy, context_level)

        chunks = self._build_chunks(arguments["file_id"], arguments["raw_markdown"], target_level)

        self.logger.info(
            "chunk_markdown: file_id=%s strategy=%s target_level=%s chunks=%d",
            arguments["file_id"],
            strategy,
            target_level,
            len(chunks),
        )

        return {"chunks": chunks}
