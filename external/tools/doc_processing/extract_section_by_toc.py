from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


class ExtractSectionByTOCTool(BaseMCPTool):
    """Extract a section using table-of-contents anchors (string search)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "extract_section_by_toc",
            "description": "Extracts a section by locating titles listed in the table of contents.",
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
                "toc_entries": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["section_title", "raw_markdown", "toc_entries"],
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
        toc_entries: List[Dict[str, Any]] = arguments.get("toc_entries", [])

        positions = self._locate_toc_matches(raw_markdown, toc_entries)
        if not positions:
            return self._empty_result(section_title, "No TOC entries found in text")

        best = self._select_entry(section_title, positions)
        if not best:
            return self._empty_result(section_title, "No TOC match for requested section")

        idx, entry, score = best
        start_char = idx
        end_char = self._find_next_anchor(idx, positions, raw_markdown)
        snippet = raw_markdown[start_char:end_char].strip()

        if not snippet:
            return self._empty_result(section_title, "Extracted text empty")

        confidence = "high" if score >= 0.8 else "medium" if score >= 0.55 else "low"

        return {
            "section_title": section_title,
            "method": "toc",
            "text": snippet,
            "confidence": confidence,
            "match_score": score,
            "page_range": [entry.get("page_number"), None],
            "char_range": [start_char, end_char],
            "issues": [],
            "notes": f"Matched TOC entry '{entry.get('title')}'",
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _locate_toc_matches(raw_markdown: str, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lower_text = raw_markdown.lower()
        positions: List[Dict[str, Any]] = []
        for entry in entries:
            title = entry.get("title") or ""
            if not title.strip():
                continue
            target = title.lower().strip()
            idx = lower_text.find(target)
            if idx == -1:
                continue
            match = dict(entry)
            match["char_index"] = idx
            positions.append(match)
        positions.sort(key=lambda e: e.get("char_index", 0))
        return positions

    def _select_entry(
        self,
        section_title: str,
        positions: List[Dict[str, Any]],
    ) -> Optional[tuple[int, Dict[str, Any], float]]:
        target = section_title.lower().strip()
        if not target:
            return None

        best_score = 0.0
        best_entry: Optional[Dict[str, Any]] = None
        for entry in positions:
            candidate = (entry.get("title") or "").lower().strip()
            if not candidate:
                continue
            score = difflib.SequenceMatcher(None, target, candidate).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry

        if not best_entry or best_score < 0.4:
            return None

        idx = best_entry.get("char_index")
        if idx is None:
            return None

        return idx, best_entry, best_score

    @staticmethod
    def _find_next_anchor(start_idx: int, positions: List[Dict[str, Any]], text: str) -> int:
        after = [entry for entry in positions if (entry.get("char_index") or 0) > start_idx]
        if after:
            return after[0]["char_index"]
        return len(text)

    def _empty_result(self, section_title: str, reason: str) -> Dict[str, Any]:
        return {
            "section_title": section_title,
            "method": "toc",
            "text": "",
            "confidence": "low",
            "match_score": 0.0,
            "page_range": [None, None],
            "char_range": [None, None],
            "issues": [reason],
            "notes": reason,
        }

