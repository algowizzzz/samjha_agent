"""Vector similarity entity-to-column mapping."""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool

MODEL_ENV_VAR = "DEEP_RESEARCH_EMBEDDING_MODEL"
DEFAULT_MODEL_NAME = os.getenv(MODEL_ENV_VAR, "sentence-transformers/all-MiniLM-L6-v2")


class VectorSearchEntityColumnTool(BaseMCPTool):
    """Compute cosine similarity between an entity value and column embeddings."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_model = self.config.get("model_name", DEFAULT_MODEL_NAME)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string", "description": "Entity text to embed"},
                "index_path": {"type": "string", "description": "Path to embeddings index JSON"},
                "entity_context": {"type": "string", "description": "Optional context to append"},
                "top_k": {"type": "integer", "description": "Number of results to return"},
                "model_name": {"type": "string", "description": "Override embedding model name"},
                "table_filter": {"type": "array", "description": "Limit search to these tables"},
            },
            "required": ["entity_value", "index_path"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string"},
                "model_used": {"type": "string"},
                "index_path": {"type": "string"},
                "top_matches": {"type": "array"},
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_arguments(arguments)

        entity_value = arguments["entity_value"]
        index_path = arguments["index_path"]
        entity_context = arguments.get("entity_context")
        top_k = max(1, int(arguments.get("top_k", 10)))
        table_filter = set(arguments.get("table_filter") or [])

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Embeddings index not found: {index_path}")

        index = self._load_index(index_path)
        model_name = arguments.get("model_name") or index.get("model") or self.default_model
        embeddings = index.get("columns", [])
        if not embeddings:
            return {
                "entity_value": entity_value,
                "model_used": model_name,
                "index_path": index_path,
                "top_matches": [],
            }

        # Optionally filter by table
        if table_filter:
            embeddings = [
                entry for entry in embeddings if entry.get("table") in table_filter
            ]

        entity_text = entity_value
        if entity_context:
            entity_text += f"\nContext: {entity_context}"

        entity_embedding = self._embed_text(model_name, entity_text)

        matches = []
        for entry in embeddings:
            column_embedding = entry.get("embedding")
            if not column_embedding:
                continue
            similarity = self._cosine_similarity(entity_embedding, column_embedding)
            matches.append(
                {
                    "table": entry.get("table"),
                    "column": entry.get("column"),
                    "data_type": entry.get("data_type"),
                    "score": round(float(similarity), 4),
                    "signature": entry.get("signature"),
                    "metadata": entry.get("metadata", {}),
                }
            )

        matches.sort(key=lambda item: item["score"], reverse=True)
        matches = matches[:top_k]

        return {
            "entity_value": entity_value,
            "model_used": model_name,
            "index_path": index_path,
            "top_matches": matches,
        }

    # Helper methods --------------------------------------------------
    def _embed_text(self, model_name: str, text: str) -> List[float]:
        model = self._get_model(model_name)
        embedding = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=False,
        )[0]
        return [float(x) for x in embedding]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        length = min(len(vec_a), len(vec_b))
        if length == 0:
            return 0.0
        dot = sum(vec_a[i] * vec_b[i] for i in range(length))
        # embeddings are normalized, but guard against drift
        mag_a = math.sqrt(sum(vec_a[i] * vec_a[i] for i in range(length)))
        mag_b = math.sqrt(sum(vec_b[i] * vec_b[i] for i in range(length)))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _load_index(index_path: str) -> Dict[str, Any]:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_model(model_name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
