from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

from tools.base_mcp_tool import BaseMCPTool


class ExtractSectionByHeadingsTool(BaseMCPTool):
    """Uses parsed markdown headings to slice a section from the document."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "extract_section_by_headings",
            "description": "Extracts a section by matching the requested title against markdown headings.",
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
                "section_title": {"type": "string"},
                "raw_markdown": {"type": "string"},
                "headings": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["section_title", "raw_markdown", "headings"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "section_title": {"type": "string"},
                "method": {"type": "string"},
                "text": {"type": "string"},
                "confidence": {"type": "string"},
                "match_score": {"type": "number"},
                "page_range": {"type": "array", "items": {"type": ["integer", "null"]}},
                "char_range": {"type": "array", "items": {"type": ["integer", "null"]}},
                "issues": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
            "required": ["section_title", "method", "text"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        section_title = arguments["section_title"]
        raw_markdown = arguments["raw_markdown"]
        headings: List[Dict[str, Any]] = arguments.get("headings", [])

        lines = raw_markdown.splitlines()
        best = self._select_heading(section_title, headings)
        if not best:
            return self._empty_result(section_title, "No heading match found")

        start_line_idx, heading_meta = best
        end_line_idx = self._find_section_end(start_line_idx, heading_meta, headings, len(lines))
        snippet = "\n".join(lines[start_line_idx:end_line_idx]).strip()

        if not snippet:
            return self._empty_result(section_title, "Extracted text empty")

        match_score = heading_meta.get("match_score", 0.0)
        confidence = "high" if match_score >= 0.8 else "medium" if match_score >= 0.55 else "low"

        char_start, char_end = self._char_range(lines, start_line_idx, end_line_idx)

        return {
            "section_title": section_title,
            "method": "headings",
            "text": snippet,
            "confidence": confidence,
            "match_score": match_score,
            "page_range": [None, None],
            "char_range": [char_start, char_end],
            "issues": [],
            "notes": f"Matched heading '{heading_meta.get('title')}' (level {heading_meta.get('level')})",
        }

    # ------------------------------------------------------------------ #
    def _select_heading(
        self, section_title: str, headings: List[Dict[str, Any]]
    ) -> Optional[Tuple[int, Dict[str, Any]]]:
        target = self._normalise(section_title)
        if not target:
            return None

        best_score = 0.0
        best_heading: Optional[Dict[str, Any]] = None
        for heading in headings:
            candidate = self._normalise(heading.get("title", ""))
            if not candidate:
                continue
            score = difflib.SequenceMatcher(None, target, candidate).ratio()
            if score > best_score:
                best_score = score
                heading["match_score"] = score
                best_heading = heading

        if not best_heading or best_score < 0.45:
            return None

        line_number = best_heading.get("line_number")
        if not line_number:
            return None

        start_line_idx = max(int(line_number) - 1, 0)
        return start_line_idx, best_heading

    def _find_section_end(
        self,
        start_line_idx: int,
        heading_meta: Dict[str, Any],
        headings: List[Dict[str, Any]],
        total_lines: int,
    ) -> int:
        current_level = self._level_to_int(heading_meta.get("level"))
        current_line_number = heading_meta.get("line_number") or 0

        for heading in headings:
            line_num = heading.get("line_number")
            if not line_num or line_num <= current_line_number:
                continue
            level = self._level_to_int(heading.get("level"))
            if level <= current_level:
                return max(int(line_num) - 1, start_line_idx + 1)

        return total_lines

    @staticmethod
    def _level_to_int(value: Optional[str]) -> int:
        if not value:
            return 99
        try:
            return int(value.lstrip("Hh"))
        except ValueError:
            return 99

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _char_range(lines: List[str], start_idx: int, end_idx: int) -> Tuple[int, int]:
        prefix = "\n".join(lines[:start_idx])
        snippet = "\n".join(lines[start_idx:end_idx])
        start_char = len(prefix)
        if prefix:
            start_char += 1  # account for newline
        end_char = start_char + len(snippet)
        return start_char, end_char

    def _empty_result(self, section_title: str, reason: str) -> Dict[str, Any]:
        return {
            "section_title": section_title,
            "method": "headings",
            "text": "",
            "confidence": "low",
            "match_score": 0.0,
            "page_range": [None, None],
            "char_range": [None, None],
            "issues": [reason],
            "notes": reason,
        }

