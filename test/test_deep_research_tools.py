import json
from pathlib import Path

import pytest

from tools.impl.pattern_match_tool import PatternMatchEntityColumnTool
from tools.impl.vector_search_tool import VectorSearchEntityColumnTool
from tools.impl.llm_map_tool import LLMMapEntityColumnTool
from tools.impl.create_column_embeddings_tool import CreateColumnEmbeddingsTool
from tools.impl.search_entity_in_data_tool import SearchEntityInDataTool


def test_pattern_match_exact_value():
    tool = PatternMatchEntityColumnTool()
    result = tool.execute(
        {
            "entity_value": "RBC",
            "columns": [
                {
                    "table": "exposure_data",
                    "column": "company_name",
                    "sample_values": ["RBC", "JPMorgan"],
                    "data_type": "VARCHAR",
                },
                {
                    "table": "limits",
                    "column": "limit_name",
                    "sample_values": ["PV01"],
                },
            ],
        }
    )

    assert result["matches"], "Expected at least one match"
    assert result["matches"][0]["column"] == "company_name"
    assert result["matches"][0]["score"] == 1.0


def test_vector_search_with_stubbed_model(tmp_path, monkeypatch):
    index_path = tmp_path / "embeddings.json"
    index_payload = {
        "model": "test-model",
        "dimension": 2,
        "columns": [
            {
                "table": "exposure_data",
                "column": "company_name",
                "data_type": "VARCHAR",
                "signature": "company field",
                "metadata": {},
                "embedding": [1.0, 0.0],
            },
            {
                "table": "limits",
                "column": "limit_name",
                "data_type": "VARCHAR",
                "signature": "limit field",
                "metadata": {},
                "embedding": [0.0, 1.0],
            },
        ],
    }
    index_path.write_text(json.dumps(index_payload))

    class DummyModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=False):
            # Always return a vector aligned with the first column
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        VectorSearchEntityColumnTool,
        "_get_model",
        staticmethod(lambda _name: DummyModel()),
    )

    tool = VectorSearchEntityColumnTool()
    result = tool.execute(
        {
            "entity_value": "RBC",
            "index_path": str(index_path),
            "top_k": 1,
        }
    )

    assert result["top_matches"], "Expected at least one vector match"
    assert result["top_matches"][0]["column"] == "company_name"


def test_llm_map_tool_fallback(monkeypatch):
    # Force heuristic fallback by disabling LLM availability
    monkeypatch.setattr(
        "tools.impl.llm_map_tool.is_llm_available",
        lambda: False,
    )

    tool = LLMMapEntityColumnTool()
    result = tool.execute(
        {
            "entity_value": "RBC",
            "candidate_columns": [
                {"table": "exposure_data", "column": "company_name", "sample_values": ["RBC"]},
                {"table": "limits", "column": "limit_name", "sample_values": ["PV01"]},
            ],
        }
    )

    assert result["selected_column"]["column"] == "company_name"
    assert result["llm_used"] is False


def test_create_column_embeddings(tmp_path, monkeypatch):
    output_path = tmp_path / "index.json"

    class DummyModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=False):
            vectors = []
            for idx, _ in enumerate(texts):
                vectors.append([float(idx), float(idx)])
            return vectors

    monkeypatch.setattr(
        CreateColumnEmbeddingsTool,
        "_get_model",
        staticmethod(lambda _name: DummyModel()),
    )

    tool = CreateColumnEmbeddingsTool()
    result = tool.execute(
        {
            "columns": [
                {
                    "table": "exposure_data",
                    "column": "company_name",
                    "data_type": "VARCHAR",
                    "sample_values": ["RBC"],
                },
                {
                    "table": "exposure_data",
                    "column": "date",
                    "data_type": "DATE",
                },
            ],
            "output_path": str(output_path),
            "model_name": "dummy-model",
            "overwrite": True,
        }
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["column_count"] == 2
    assert result["index_path"] == str(output_path)


def test_search_entity_in_data(tmp_path):
    csv_path = tmp_path / "exposure.csv"
    csv_path.write_text("company_name,desk\nRBC,Americas\nJPMorgan,EMEA\nRoyal Bank of Canada,Global\n")

    tool = SearchEntityInDataTool()
    result = tool.execute(
        {
            "entity_value": "RBC",
            "tables": [
                {
                    "name": "exposure_data",
                    "path": str(csv_path),
                    "format": "csv",
                    "text_columns": ["company_name", "desk"],
                }
            ],
            "max_results_per_column": 5,
        }
    )

    assert result["total_matches"] >= 1
    table_match = next(
        match
        for match in result["matches"]
        if match.get("column") == "company_name"
    )
    assert any("RBC" in str(value) for value in table_match.get("sample_values", []))
