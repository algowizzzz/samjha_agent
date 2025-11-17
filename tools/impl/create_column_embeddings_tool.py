"""Generate embeddings for table columns."""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool

MODEL_ENV_VAR = "DEEP_RESEARCH_EMBEDDING_MODEL"
DEFAULT_MODEL_NAME = os.getenv(MODEL_ENV_VAR, "sentence-transformers/all-MiniLM-L6-v2")


class CreateColumnEmbeddingsTool(BaseMCPTool):
    """Compute embeddings for discovered table columns and persist them as an index."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.default_model = self.config.get("model_name", DEFAULT_MODEL_NAME)
        self.default_output_dir = self.config.get(
            "output_dir", os.path.join("data", "agent_state")
        )

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "description": "List of column metadata records",
                },
                "model_name": {
                    "type": "string",
                    "description": "Sentence-transformer model to use",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path for the embeddings index",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Allow overwriting existing index file",
                },
            },
            "required": ["columns"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index_path": {"type": "string"},
                "model": {"type": "string"},
                "dimension": {"type": "integer"},
                "column_count": {"type": "integer"},
                "created_at": {"type": "string"},
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_arguments(arguments)

        columns = [
            column for column in arguments.get("columns", []) if isinstance(column, dict)
        ]
        if not columns:
            raise ValueError("No columns supplied for embedding generation.")

        model_name = arguments.get("model_name") or self.default_model
        output_path = arguments.get("output_path") or self._default_output_path()
        overwrite = bool(arguments.get("overwrite", False))

        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                f"Embeddings index already exists at {output_path}. "
                "Pass overwrite=true to replace it."
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        model = self._get_model(model_name)
        signatures = [self._build_signature(column) for column in columns]
        embeddings = model.encode(
            [signature["text"] for signature in signatures],
            normalize_embeddings=True,
            convert_to_numpy=False,
        )

        serialized_columns: List[Dict[str, Any]] = []
        for signature, embedding in zip(signatures, embeddings):
            serialized_columns.append(
                {
                    "table": signature["table"],
                    "column": signature["column"],
                    "data_type": signature["data_type"],
                    "signature": signature["text"],
                    "metadata": signature["metadata"],
                    "embedding": [float(value) for value in embedding],
                }
            )

        index_payload = {
            "model": model_name,
            "dimension": len(serialized_columns[0]["embedding"])
            if serialized_columns
            else 0,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "column_count": len(serialized_columns),
            "columns": serialized_columns,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(index_payload, f, ensure_ascii=False, indent=2)

        return {
            "index_path": output_path,
            "model": model_name,
            "dimension": index_payload["dimension"],
            "column_count": len(serialized_columns),
            "created_at": index_payload["created_at"],
        }

    # Helpers ----------------------------------------------------------
    def _default_output_path(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"deep_research_embeddings_{timestamp}.json"
        return os.path.join(self.default_output_dir, filename)

    @staticmethod
    def _build_signature(column: Dict[str, Any]) -> Dict[str, Any]:
        table = str(column.get("table") or "")
        col_name = str(column.get("column") or "")
        data_type = column.get("data_type") or column.get("type")
        description = column.get("column_description") or column.get("description") or ""
        table_description = column.get("table_description") or ""
        sample_values = column.get("sample_values") or []
        tags = column.get("tags") or []

        signature_parts = [
            f"Table: {table}",
            f"Column: {col_name}",
            f"Type: {data_type}",
            f"Description: {description}",
        ]
        if table_description:
            signature_parts.append(f"Table Description: {table_description}")
        if sample_values:
            preview = ", ".join(map(str, sample_values[:10]))
            signature_parts.append(f"Samples: {preview}")
        if tags:
            signature_parts.append(f"Tags: {', '.join(map(str, tags))}")

        signature_text = "\n".join(signature_parts)
        metadata = {
            "table_description": table_description,
            "sample_values": sample_values[:25],
            "tags": tags,
        }

        return {
            "table": table,
            "column": col_name,
            "data_type": data_type,
            "text": signature_text,
            "metadata": metadata,
        }

    @staticmethod
    @lru_cache(maxsize=2)
    def _get_model(model_name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
