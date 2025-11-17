from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Tuple

from tools.base_mcp_tool import BaseMCPTool

# Fuzzy matching threshold (0.0 to 1.0)
FUZZY_MATCH_THRESHOLD = 0.95


class ApplyChangesDeterministicTool(BaseMCPTool):
    """Applies suggested changes by performing exact string replacements."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "apply_changes_deterministic",
            "description": "Applies suggested changes using exact substring replacement to avoid hallucinations.",
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
                "raw_markdown": {"type": "string"},
                "changes": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["raw_markdown", "changes"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "new_raw_markdown": {"type": "string"},
                "applied_change_ids": {"type": "array", "items": {"type": "string"}},
                "failed_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                        "required": ["id", "reason"],
                    },
                },
            },
            "required": ["new_raw_markdown", "applied_change_ids", "failed_changes"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        working_text = arguments["raw_markdown"]
        applied: List[str] = []
        failed: List[Dict[str, str]] = []

        # Track offset adjustments from previous replacements
        offset_adjustment = 0

        for change in arguments.get("changes", []):
            change_id = change.get("id") or f"CHG-{len(applied) + len(failed) + 1:03d}"
            original = change.get("original_text") or ""
            replacement = change.get("suggested_text") or ""
            char_range = change.get("char_range")  # [start, end]

            if not original.strip():
                failed.append({"id": change_id, "reason": "original_text missing"})
                continue

            # Strategy 1: Use char_range if available (most precise)
            if char_range and isinstance(char_range, list) and len(char_range) == 2:
                result = self._replace_by_char_range(
                    working_text, char_range, original, replacement, offset_adjustment
                )
                if result:
                    working_text, offset_delta = result
                    offset_adjustment += offset_delta
                    applied.append(change_id)
                    continue
                # Fallthrough to text search if char_range method failed

            # Strategy 2: Exact text match
            result = self._replace_once(working_text, original, replacement)
            if result is not None:
                working_text = result
                applied.append(change_id)
                continue

            # Strategy 3: Fuzzy text match (95% similarity)
            result = self._replace_fuzzy(working_text, original, replacement)
            if result is not None:
                working_text = result
                applied.append(change_id)
                self.logger.info(f"Applied {change_id} using fuzzy match")
                continue

            # All strategies failed
            failed.append({"id": change_id, "reason": "no match found (exact, fuzzy, or char_range)"})

        return {
            "new_raw_markdown": working_text,
            "applied_change_ids": applied,
            "failed_changes": failed,
        }

    @staticmethod
    def _replace_once(text: str, original: str, replacement: str) -> Optional[str]:
        count = text.count(original)
        if count != 1:
            return None
        return text.replace(original, replacement, 1)

    @staticmethod
    def _replace_by_char_range(
        text: str, char_range: List[int], expected_original: str, replacement: str, offset: int
    ) -> Optional[Tuple[str, int]]:
        """
        Replace text using character range positions.
        Returns (new_text, offset_delta) or None if validation fails.
        """
        start, end = char_range
        # Adjust for cumulative offset from previous replacements
        start += offset
        end += offset

        # Validate boundaries
        if start < 0 or end > len(text) or start >= end:
            return None

        # Extract actual text at position
        actual_text = text[start:end]

        # Validate that actual text roughly matches expected (allow some whitespace variation)
        normalized_actual = " ".join(actual_text.split())
        normalized_expected = " ".join(expected_original.split())

        similarity = difflib.SequenceMatcher(None, normalized_actual, normalized_expected).ratio()
        if similarity < 0.85:  # Lower threshold for char_range since position is trusted
            return None

        # Perform replacement
        new_text = text[:start] + replacement + text[end:]

        # Calculate offset delta for subsequent replacements
        offset_delta = len(replacement) - (end - start)

        return (new_text, offset_delta)

    def _replace_fuzzy(self, text: str, original: str, replacement: str) -> Optional[str]:
        """
        Find and replace text using fuzzy matching.
        Returns new text if exactly one high-confidence match found, else None.
        """
        normalized_original = self._normalize_text(original)

        # Sliding window to find best match
        window_size = len(original)
        best_match_pos = -1
        best_match_score = 0.0

        for i in range(len(text) - window_size + 1):
            candidate = text[i : i + window_size]
            normalized_candidate = self._normalize_text(candidate)

            score = difflib.SequenceMatcher(None, normalized_original, normalized_candidate).ratio()

            if score > best_match_score:
                best_match_score = score
                best_match_pos = i

        # Require high confidence match
        if best_match_score < FUZZY_MATCH_THRESHOLD:
            return None

        # Check for ambiguity - ensure only one match above threshold
        matches_above_threshold = 0
        for i in range(len(text) - window_size + 1):
            if i == best_match_pos:
                continue
            candidate = text[i : i + window_size]
            normalized_candidate = self._normalize_text(candidate)
            score = difflib.SequenceMatcher(None, normalized_original, normalized_candidate).ratio()
            if score >= FUZZY_MATCH_THRESHOLD:
                matches_above_threshold += 1

        if matches_above_threshold > 0:
            return None  # Ambiguous, multiple matches

        # Apply replacement
        return text[:best_match_pos] + replacement + text[best_match_pos + window_size :]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace for fuzzy comparison."""
        return " ".join(text.split()).lower()

