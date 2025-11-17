import json
from pathlib import Path

import pytest

from external.agent.deep_research.deep_research_agent import DeepResearchAgent
from tools.impl.create_column_embeddings_tool import CreateColumnEmbeddingsTool
from tools.impl.vector_search_tool import VectorSearchEntityColumnTool


@pytest.fixture(scope="function")
def deep_research_agent(tmp_path, monkeypatch):
    # Reset singleton config state for isolated tests
    monkeypatch.setattr(
        "external.agent.deep_research.deep_research_config.DeepResearchAgentConfig._instance",
        None,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Exposure data
    (data_dir / "exposure_data.csv").write_text(
        "company,date,exposure_amt,region\n"
        "RBC,2024-11-14,100,Canada\n"
        "RBC,2024-11-13,95,Canada\n"
        "ACME,2024-11-14,50,US\n"
    )

    # Breach data
    (data_dir / "breach_data.csv").write_text(
        "company,limit_name,breach_date\n"
        "RBC,Limit A,2024-11-14\n"
        "ACME,Limit B,2024-11-13\n"
    )

    # Trades data
    (data_dir / "trade_data.csv").write_text(
        "trade_id,company,trade_date,notional\n"
        "T1,RBC,2024-11-14,1000\n"
        "T2,ACME,2024-11-14,500\n"
    )

    # Financial statements
    (data_dir / "income_statement.csv").write_text(
        "company,period,revenue,expenses\n"
        "ACME,2024Q3,1000,700\n"
        "ACME,2024Q2,900,650\n"
    )
    (data_dir / "balance_sheet.csv").write_text(
        "company,period,total_assets,total_liabilities\n"
        "ACME,2024Q3,5000,3000\n"
        "ACME,2024Q2,4800,2900\n"
    )
    (data_dir / "projects.csv").write_text(
        "project_name,status,owner\n"
        "Phoenix,On Track,RBC\n"
        "Helios,Delayed,ACME\n"
    )

    class DummyModel:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=False):
            vectors = []
            for idx, _ in enumerate(texts):
                vectors.append([float(idx + 1), float(idx + 2)])
            return vectors

    monkeypatch.setattr(
        CreateColumnEmbeddingsTool,
        "_get_model",
        staticmethod(lambda _name: DummyModel()),
    )
    monkeypatch.setattr(
        VectorSearchEntityColumnTool,
        "_get_model",
        staticmethod(lambda _name: DummyModel()),
    )

    config = {
        "data": {
            "search_roots": [str(data_dir)],
            "extensions": [".csv"],
            "text_value_limit": 50,
        },
        "query": {"default_limit": 50},
        "tool_loop": {"max_turns": 2},
        "prompts": {},
        "llm": {"provider": "none", "model": "none", "temperature": 0},
    }

    config_path = tmp_path / "deep_research_config.json"
    config_path.write_text(json.dumps(config))

    agent = DeepResearchAgent(config_path=str(config_path))
    return agent


def test_single_table_query(deep_research_agent):
    agent = deep_research_agent
    state = agent.run_query("What was the exposure for RBC yesterday?")

    assert state["sql_queries"], "Expected at least one SQL query"
    primary_query = state["sql_queries"][0]
    assert "exposure_data" in primary_query["sql"], "Exposure table should be referenced"
    assert "rbc" in primary_query["sql"].lower()


def test_multi_table_decomposition(deep_research_agent):
    agent = deep_research_agent
    state = agent.run_query("Show trades related to RBC breach details")

    tables = {query["table"] for query in state.get("sql_queries", [])}
    assert len(tables) >= 2, "Expected multiple tables in decomposition"
    assert "trade_data" in tables
    assert "breach_data" in tables or "exposure_data" in tables


def test_financial_analysis_requires_parallel_queries(deep_research_agent):
    agent = deep_research_agent
    state = agent.run_query("Give me a financial analysis of ACME")

    tables = {query["table"] for query in state.get("sql_queries", [])}
    assert "income_statement" in tables
    assert "balance_sheet" in tables


def test_ambiguous_entity_generates_reasoning(deep_research_agent):
    agent = deep_research_agent
    state = agent.run_query("What is the status of project Phoenix?")

    reasoning = state.get("analytical_reasoning", {})
    assert reasoning, "Analytical reasoning should be captured"
    assert "projects" in json.dumps(reasoning)


def test_entity_not_found_reports_zero_matches(deep_research_agent):
    agent = deep_research_agent
    state = agent.run_query("Find details for NonExistent Company")

    entity_search = state.get("entity_search_results", {})
    assert entity_search, "Entity search results should not be empty"
    first_key = next(iter(entity_search))
    values = entity_search[first_key]
    assert values.get("total_matches", 0) == 0
