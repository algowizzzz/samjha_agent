"""LLM-assisted entity-to-column mapper with heuristic fallback."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool

try:
    from external.agent.llm_client import get_llm_client, is_llm_available
except ImportError:  # pragma: no cover
    get_llm_client = None

    def is_llm_available() -> bool:  # type: ignore
        return False


class LLMMapEntityColumnTool(BaseMCPTool):
    """
    Uses an LLM to choose the best column match for an entity.
    Falls back to heuristics when LLM access is unavailable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string"},
                "candidate_columns": {"type": "array"},
                "entity_context": {"type": "string"},
                "instructions": {"type": "string"},
            },
            "required": ["entity_value", "candidate_columns"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string"},
                "llm_used": {"type": "boolean"},
                "selected_column": {"type": "object"},
                "alternatives": {"type": "array"},
                "reasoning": {"type": "string"},
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_arguments(arguments)

        entity_value = arguments["entity_value"]
        candidates = [
            candidate
            for candidate in arguments.get("candidate_columns", [])
            if isinstance(candidate, dict)
        ]
        entity_context = arguments.get("entity_context", "")
        instructions = arguments.get("instructions")

        if not candidates:
            return {
                "entity_value": entity_value,
                "llm_used": False,
                "selected_column": None,
                "alternatives": [],
                "reasoning": "No candidate columns were provided.",
            }

        if is_llm_available():
            try:
                result = self._run_llm_mapping(
                    entity_value,
                    candidates,
                    entity_context=entity_context,
                    instructions=instructions,
                )
                result["llm_used"] = True
                return result
            except Exception as exc:  # pragma: no cover - fallback path
                reasoning = f"LLM mapping failed: {exc}. Falling back to heuristics."
        else:
            reasoning = "LLM unavailable; using heuristic mapping."

        heuristic_result = self._heuristic_mapping(entity_value, candidates, reasoning)
        heuristic_result["llm_used"] = False
        return heuristic_result

    # Internal helpers -------------------------------------------------
    def _run_llm_mapping(
        self,
        entity_value: str,
        candidates: List[Dict[str, Any]],
        entity_context: str = "",
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the LLM to select the best column."""
        llm = get_llm_client()
        if llm is None or not llm.is_available():
            raise RuntimeError("LLM client not available")

        system_prompt = (
            "You map user entities to structured database columns. "
            "Return valid JSON with keys: selected_column (table, column, confidence, reasoning) "
            "and alternatives (array of similar entries).\n"
        )
        if instructions:
            system_prompt += f"\nAdditional instructions: {instructions}\n"

        candidate_text = self._format_candidates(candidates)
        user_prompt = (
            f"Entity: {entity_value}\n"
            f"Context: {entity_context or 'N/A'}\n\n"
            f"Candidate Columns:\n{candidate_text}\n"
            "Respond with JSON."
        )

        raw_response = llm.invoke_with_prompt(
            system_prompt, user_prompt, response_format="json"
        )
        parsed = self._safe_json_loads(raw_response)
        selected = parsed.get("selected_column") or {}
        alternatives = parsed.get("alternatives") or []

        return {
            "entity_value": entity_value,
            "selected_column": selected,
            "alternatives": alternatives,
            "reasoning": selected.get("reasoning", ""),
        }

    def _heuristic_mapping(
        self,
        entity_value: str,
        candidates: List[Dict[str, Any]],
        reasoning: str,
    ) -> Dict[str, Any]:
        """Fallback deterministic mapping."""
        entity_tokens = set(entity_value.lower().split())

        def score_candidate(candidate: Dict[str, Any]) -> float:
            column_name = str(candidate.get("column", "")).lower()
            table_name = str(candidate.get("table", "")).lower()
            sample_values = [
                str(val).lower() for val in candidate.get("sample_values", [])[:50]
            ]

            score = 0.0
            if entity_value.lower() in column_name:
                score += 0.6
            if entity_value.lower() in table_name:
                score += 0.4
            for value in sample_values:
                if value == entity_value.lower():
                    score += 1.0
                    break
                if entity_value.lower() in value or value in entity_value.lower():
                    score += 0.5
                    break

            column_tokens = set(column_name.split())
            overlap = entity_tokens.intersection(column_tokens)
            if overlap:
                score += 0.3 * (len(overlap) / max(len(entity_tokens), 1))

            return score

        scored = [
            (score_candidate(candidate), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        best_score, best_candidate = scored[0]
        alternatives = [
            {
                "table": cand.get("table"),
                "column": cand.get("column"),
                "score": round(score, 4),
            }
            for score, cand in scored[1:3]
        ]

        selected = {
            "table": best_candidate.get("table"),
            "column": best_candidate.get("column"),
            "confidence": round(min(1.0, max(0.0, best_score)), 4),
            "reasoning": reasoning,
        }

        return {
            "entity_value": entity_value,
            "selected_column": selected,
            "alternatives": alternatives,
            "reasoning": reasoning,
        }

    @staticmethod
    def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
        """Create readable text for LLM prompt."""
        lines = []
        for idx, candidate in enumerate(candidates, start=1):
            samples = candidate.get("sample_values") or []
            sample_preview = ", ".join(map(str, samples[:5]))
            lines.append(
                f"{idx}. Table={candidate.get('table')} | Column={candidate.get('column')} | "
                f"Type={candidate.get('data_type')} | Samples=[{sample_preview}] | "
                f"Description={candidate.get('description')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _safe_json_loads(raw: str) -> Dict[str, Any]:
        """Parse JSON, stripping code fences if necessary."""
        text = raw.strip()
        if text.startswith("```"):
            # Remove potential ```json ... ``` wrappers
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("LLM response was not valid JSON.")
