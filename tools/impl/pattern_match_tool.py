"""Pattern-based entity-to-column matcher."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


def _normalize(text: Optional[str]) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    return text.strip().lower()


def _tokenize(text: str) -> List[str]:
    """Split text into alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class ColumnCandidate:
    table: str
    column: str
    data_type: Optional[str]
    sample_values: List[str]
    description: Optional[str]
    table_description: Optional[str]


class PatternMatchEntityColumnTool(BaseMCPTool):
    """
    Heuristic matcher that scores columns based on lexical similarity
    between an entity value and column metadata/sample values.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    # BaseMCPTool requires schema accessors
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string"},
                "columns": {"type": "array"},
                "max_results": {"type": "integer"},
                "case_sensitive": {"type": "boolean"},
            },
            "required": ["entity_value", "columns"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string"},
                "matches": {"type": "array"},
                "max_score": {"type": "number"},
                "total_matches": {"type": "integer"},
            },
        }

    # Core execution -----------------------------------------------------
    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_arguments(arguments)

        entity_value = arguments["entity_value"]
        max_results = max(1, int(arguments.get("max_results", 10)))
        case_sensitive = bool(arguments.get("case_sensitive", False))

        candidates = [
            self._parse_candidate(raw_candidate)
            for raw_candidate in arguments["columns"]
            if isinstance(raw_candidate, dict)
        ]

        if not candidates:
            return {
                "entity_value": entity_value,
                "matches": [],
                "max_score": 0.0,
                "total_matches": 0,
            }

        matches = []
        entity_norm = entity_value if case_sensitive else _normalize(entity_value)
        entity_tokens = _tokenize(entity_value)

        for candidate in candidates:
            score, match_type, evidence = self._score_candidate(
                entity_value,
                entity_norm,
                entity_tokens,
                candidate,
                case_sensitive,
            )
            if score > 0:
                matches.append(
                    {
                        "table": candidate.table,
                        "column": candidate.column,
                        "data_type": candidate.data_type,
                        "score": round(score, 4),
                        "match_type": match_type,
                        "evidence": evidence,
                        "description": candidate.description,
                        "table_description": candidate.table_description,
                    }
                )

        matches.sort(key=lambda item: item["score"], reverse=True)
        top_matches = matches[:max_results]
        max_score = top_matches[0]["score"] if top_matches else 0.0

        return {
            "entity_value": entity_value,
            "matches": top_matches,
            "max_score": max_score,
            "total_matches": len(matches),
        }

    # Helpers ------------------------------------------------------------
    def _parse_candidate(self, data: Dict[str, Any]) -> ColumnCandidate:
        return ColumnCandidate(
            table=str(data.get("table") or ""),
            column=str(data.get("column") or ""),
            data_type=data.get("data_type"),
            sample_values=[
                str(value) for value in data.get("sample_values", [])[:50]
            ],
            description=data.get("column_description") or data.get("description"),
            table_description=data.get("table_description"),
        )

    def _score_candidate(
        self,
        entity_value: str,
        entity_norm: str,
        entity_tokens: List[str],
        candidate: ColumnCandidate,
        case_sensitive: bool,
    ) -> tuple[float, str, List[str]]:
        """Compute heuristic score for a column candidate."""
        best_score = 0.0
        best_type = "none"
        evidence: List[str] = []

        col_name = candidate.column if case_sensitive else _normalize(candidate.column)
        table_name = candidate.table if case_sensitive else _normalize(candidate.table)

        # 1) Exact sample value match (highest confidence)
        for value in candidate.sample_values:
            cmp_value = value if case_sensitive else _normalize(value)
            if cmp_value and cmp_value == entity_norm:
                return 1.0, "value_exact", [value]

        # 2) Substring match inside sample values
        for value in candidate.sample_values:
            cmp_value = value if case_sensitive else _normalize(value)
            if not cmp_value:
                continue
            if entity_norm and entity_norm in cmp_value:
                best_score = max(best_score, 0.9)
                best_type = "value_substring"
                evidence = [value]
                break

        # 3) Column name similarity
        if entity_norm and entity_norm == col_name:
            best_score = max(best_score, 0.85)
            best_type = "column_exact"
            evidence = [candidate.column]
        elif entity_norm and entity_norm in col_name:
            best_score = max(best_score, 0.75)
            best_type = "column_substring"
            evidence = [candidate.column]

        # 4) Token overlap between entity and column/table names
        col_tokens = _tokenize(candidate.column)
        table_tokens = _tokenize(candidate.table)
        token_overlap = self._token_overlap(entity_tokens, col_tokens + table_tokens)
        if token_overlap > 0:
            score = 0.6 + 0.2 * token_overlap  # up to 0.8
            if score > best_score:
                best_score = score
                best_type = "token_overlap"
                evidence = [" ".join(col_tokens)]

        # 5) Sample value prefix / suffix similarity
        for value in candidate.sample_values:
            cmp_value = value if case_sensitive else _normalize(value)
            if not cmp_value:
                continue
            if cmp_value.startswith(entity_norm) or entity_norm.startswith(cmp_value):
                best_score = max(best_score, 0.7)
                best_type = "value_prefix_suffix"
                evidence = [value]
                break

        return best_score, best_type, evidence

    @staticmethod
    def _token_overlap(entity_tokens: List[str], column_tokens: List[str]) -> float:
        """Compute ratio of shared tokens."""
        if not entity_tokens or not column_tokens:
            return 0.0
        entity_set = set(entity_tokens)
        column_set = set(column_tokens)
        overlap = entity_set.intersection(column_set)
        if not overlap:
            return 0.0
        return len(overlap) / math.sqrt(len(entity_set) * len(column_set))
