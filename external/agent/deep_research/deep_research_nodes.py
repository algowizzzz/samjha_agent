"""LangGraph-style nodes for the Deep Research Agent."""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb

from external.agent.deep_research.deep_research_schemas import AgentState, PartialState
from tools.impl.create_column_embeddings_tool import CreateColumnEmbeddingsTool
from tools.impl.llm_map_tool import LLMMapEntityColumnTool
from tools.impl.pattern_match_tool import PatternMatchEntityColumnTool
from tools.impl.search_entity_in_data_tool import SearchEntityInDataTool
from tools.impl.vector_search_tool import VectorSearchEntityColumnTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _make_node_result(
    state: AgentState,
    node_name: str,
    control: str,
    reasoning: str,
    node_output: Optional[Dict[str, Any]] = None,
    state_updates: Optional[Dict[str, Any]] = None,
) -> PartialState:
    logs = state.get("logs", [])
    logs.append(
        {
            "node": node_name,
            "timestamp": _now_iso(),
            "msg": reasoning,
            "control": control,
        }
    )

    result: PartialState = {
        "last_node": node_name,
        "control": control,
        "node_reasoning": reasoning,
        "logs": logs,
    }
    if node_output is not None:
        result["node_output"] = node_output
    if state_updates:
        result.update(state_updates)
    return result


def _is_text_type(col_type: Optional[str]) -> bool:
    if not col_type:
        return True
    text_markers = ["char", "text", "string"]
    lowered = col_type.lower()
    return any(marker in lowered for marker in text_markers)


def _read_csv_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def _sample_csv_column_values(path: str, column_index: int, limit: int = 200) -> List[str]:
    values: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if column_index < len(row):
                value = row[column_index].strip()
                if value:
                    values.append(value)
            if len(values) >= limit:
                break
    return list(dict.fromkeys(values))  # preserve order, remove duplicates


def _sample_parquet_column_values(path: str, column: str, limit: int = 200) -> List[str]:
    query = f"SELECT DISTINCT {column} FROM read_parquet('{path}') WHERE {column} IS NOT NULL LIMIT {limit}"
    try:
        result = duckdb.query(query).fetchall()
        return [str(row[0]) for row in result if row and row[0] is not None]
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Parquet sampling failed for %s.%s: %s", path, column, exc)
        return []


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def discover_sources_node(state: AgentState, cfg: Any) -> PartialState:
    search_roots = cfg.get_nested("data", "search_roots", default=["data/duckdb"])
    allowed_extensions = cfg.get_nested("data", "extensions", default=[".csv", ".parquet"])

    discovered: List[Dict[str, Any]] = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for ext in allowed_extensions:
            pattern = os.path.join(root, f"*{ext}")
            for path in glob.glob(pattern):
                table_name = os.path.splitext(os.path.basename(path))[0]
                fmt = ext.replace(".", "").lower()
                try:
                    size_bytes = os.path.getsize(path)
                except OSError:
                    size_bytes = 0
                discovered.append(
                    {
                        "name": table_name,
                        "path": path,
                        "format": fmt,
                        "size_bytes": size_bytes,
                        "root": root,
                    }
                )

    reasoning = f"Discovered {len(discovered)} structured data sources"
    return _make_node_result(
        state,
        node_name="discover_sources",
        control="extract_schemas",
        reasoning=reasoning,
        state_updates={
            "data_sources": discovered,
            "logs": state.get("logs", []),
        },
    )


def extract_schemas_node(state: AgentState, cfg: Any) -> PartialState:
    table_schemas: Dict[str, Any] = {}
    docs_meta: List[Dict[str, Any]] = []

    for source in state.get("data_sources", []):
        table_name = source["name"]
        path = source["path"]
        fmt = source.get("format", "csv").lower()
        columns: List[Dict[str, Any]] = []

        try:
            if fmt == "csv":
                header = _read_csv_header(path)
                columns = [{"name": col, "type": "VARCHAR"} for col in header]
            else:  # parquet or other formats handled by DuckDB
                describe_query = (
                    f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
                    if path.endswith(".parquet")
                    else f"DESCRIBE SELECT * FROM read_csv_auto('{path}') LIMIT 0"
                )
                describe_rows = duckdb.query(describe_query).fetchall()
                columns = [
                    {"name": row[0], "type": row[1]}
                    for row in describe_rows
                    if row and row[0] != ""
                ]
        except Exception as exc:
            logger.warning("Schema extraction failed for %s: %s", table_name, exc)
            continue

        table_schemas[table_name] = {
            "path": path,
            "format": fmt,
            "columns": columns,
        }

        docs_meta.append(
            {
                "table": table_name,
                "description": f"Discovered from {os.path.basename(path)}",
                "column_count": len(columns),
            }
        )

    reasoning = f"Extracted schema metadata for {len(table_schemas)} tables"
    return _make_node_result(
        state,
        node_name="extract_schemas",
        control="extract_text_values",
        reasoning=reasoning,
        state_updates={
            "table_schemas": table_schemas,
            "docs_meta": docs_meta,
        },
    )


def extract_text_values_node(state: AgentState, cfg: Any) -> PartialState:
    text_values: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    limit = cfg.get_nested("data", "text_value_limit", default=200)

    for table, schema in state.get("table_schemas", {}).items():
        path = schema.get("path")
        fmt = schema.get("format", "csv")
        columns = schema.get("columns", [])
        if not path:
            continue

        for column in columns:
            col_name = column.get("name")
            if not col_name or not _is_text_type(column.get("type")):
                continue

            try:
                if fmt == "csv":
                    header = [col.get("name") for col in columns]
                    try:
                        idx = header.index(col_name)
                    except ValueError:
                        continue
                    values = _sample_csv_column_values(path, idx, limit=limit)
                else:
                    values = _sample_parquet_column_values(path, col_name, limit=limit)
            except Exception as exc:
                logger.warning("Value sampling failed for %s.%s: %s", table, col_name, exc)
                values = []

            if values:
                text_values[table][col_name] = values

    reasoning = "Collected text samples for entity mapping"
    return _make_node_result(
        state,
        node_name="extract_text_values",
        control="extract_entities",
        reasoning=reasoning,
        state_updates={"text_column_values": text_values},
    )


def extract_entities_node(state: AgentState, cfg: Any) -> PartialState:
    query = state.get("user_input", "")
    entities: List[Dict[str, Any]] = []

    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", query)
    for parts in quoted:
        text = next((p for p in parts if p), None)
        if text:
            entities.append({"text": text, "type": "quoted_phrase", "confidence": 0.95})

    upper_tokens = re.findall(r"\b[A-Z]{2,}[A-Za-z0-9]*\b", query)
    for token in upper_tokens:
        entities.append({"text": token, "type": "proper_noun", "confidence": 0.8})

    date_patterns = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query)
    for date in date_patterns:
        entities.append({"text": date, "type": "date", "confidence": 0.9})

    # Fallback to entire query keyword if no entity found
    if not entities and query:
        entities.append({"text": query, "type": "query", "confidence": 0.5})

    reasoning = f"Extracted {len(entities)} entity candidates"
    return _make_node_result(
        state,
        node_name="extract_entities",
        control="capture_intent",
        reasoning=reasoning,
        state_updates={"extracted_entities": entities},
    )


def capture_intent_node(state: AgentState, cfg: Any) -> PartialState:
    query = state.get("user_input", "")
    entities = state.get("extracted_entities", [])
    objective = "Analyze structured data"
    goal_type = "descriptive"

    if "trend" in query.lower():
        goal_type = "trend"
    if "compare" in query.lower() or "vs" in query.lower():
        goal_type = "comparative"
    if any(ent.get("type") == "date" for ent in entities):
        objective = "Time-bound analysis"

    intent = {
        "primary_objective": objective,
        "goal_type": goal_type,
        "required_entities": [ent.get("text") for ent in entities],
        "expected_outputs": ["sql_queries", "analytical_reasoning"],
        "confidence": 0.7,
    }

    reasoning = "Captured user intent snapshot"
    return _make_node_result(
        state,
        node_name="capture_intent",
        control="create_embeddings",
        reasoning=reasoning,
        state_updates={"intent_snapshot": intent},
    )


def create_embeddings_node(state: AgentState, cfg: Any) -> PartialState:
    columns_payload: List[Dict[str, Any]] = []
    text_values = state.get("text_column_values", {})

    for table, schema in state.get("table_schemas", {}).items():
        for column in schema.get("columns", []):
            col_name = column.get("name")
            if not col_name:
                continue
            columns_payload.append(
                {
                    "table": table,
                    "column": col_name,
                    "data_type": column.get("type"),
                    "sample_values": text_values.get(table, {}).get(col_name, [])[:20],
                }
            )

    if not columns_payload:
        reasoning = "No columns available for embeddings"
        return _make_node_result(
            state,
            node_name="create_embeddings",
            control="map_entities",
            reasoning=reasoning,
        )

    tool = CreateColumnEmbeddingsTool()
    result = tool.execute({"columns": columns_payload, "overwrite": True})
    reasoning = f"Embeddings index created at {result['index_path']}"
    return _make_node_result(
        state,
        node_name="create_embeddings",
        control="map_entities",
        reasoning=reasoning,
        state_updates={"embeddings_index_path": result["index_path"]},
    )


def map_entities_node(state: AgentState, cfg: Any) -> PartialState:
    pattern_tool = PatternMatchEntityColumnTool()
    vector_tool = VectorSearchEntityColumnTool()
    llm_tool = LLMMapEntityColumnTool()

    columns_metadata: List[Dict[str, Any]] = []
    text_values = state.get("text_column_values", {})
    for table, schema in state.get("table_schemas", {}).items():
        for column in schema.get("columns", []):
            columns_metadata.append(
                {
                    "table": table,
                    "column": column.get("name"),
                    "data_type": column.get("type"),
                    "sample_values": text_values.get(table, {}).get(column.get("name"), [])[:20],
                }
            )

    entity_mappings: Dict[str, Any] = {}
    embeddings_path = state.get("embeddings_index_path")

    for entity in state.get("extracted_entities", []):
        entity_value = entity.get("text", "")
        if not entity_value:
            continue

        pattern_result = pattern_tool.execute(
            {
                "entity_value": entity_value,
                "columns": columns_metadata,
                "max_results": 5,
            }
        )

        if embeddings_path:
            vector_result = vector_tool.execute(
                {
                    "entity_value": entity_value,
                    "index_path": embeddings_path,
                    "top_k": 5,
                }
            )
        else:
            vector_result = {"top_matches": []}

        candidate_subset = [
            {"table": match.get("table"), "column": match.get("column"), "sample_values": []}
            for match in pattern_result.get("matches", [])[:3]
        ]
        llm_result = llm_tool.execute(
            {
                "entity_value": entity_value,
                "candidate_columns": candidate_subset or columns_metadata[:3],
            }
        )

        entity_mappings[entity_value] = {
            "pattern_match": pattern_result,
            "vector_search": vector_result,
            "llm_mapping": llm_result,
            "final_mapping": llm_result.get("selected_column"),
        }

    reasoning = f"Mapped {len(entity_mappings)} entities to candidate columns"
    return _make_node_result(
        state,
        node_name="map_entities",
        control="search_entities",
        reasoning=reasoning,
        state_updates={"entity_mappings": entity_mappings},
    )


def search_entities_node(state: AgentState, cfg: Any) -> PartialState:
    search_tool = SearchEntityInDataTool()
    tables_payload: List[Dict[str, Any]] = []

    for table, schema in state.get("table_schemas", {}).items():
        text_columns = [
            column.get("name")
            for column in schema.get("columns", [])
            if _is_text_type(column.get("type"))
        ]
        if not text_columns:
            continue
        tables_payload.append(
            {
                "name": table,
                "path": schema.get("path"),
                "format": schema.get("format"),
                "text_columns": text_columns,
            }
        )

    entity_search_results: Dict[str, Any] = {}
    for entity in state.get("extracted_entities", []):
        entity_value = entity.get("text", "")
        if not entity_value:
            continue
        try:
            result = search_tool.execute(
                {
                    "entity_value": entity_value,
                    "tables": tables_payload,
                    "max_results_per_column": 5,
                }
            )
            entity_search_results[entity_value] = result
        except Exception as exc:
            logger.warning("Entity search failed for %s: %s", entity_value, exc)
            entity_search_results[entity_value] = {"error": str(exc)}

    reasoning = "Indexed entity occurrences across data files"
    return _make_node_result(
        state,
        node_name="search_entities",
        control="decompose_query",
        reasoning=reasoning,
        state_updates={"entity_search_results": entity_search_results},
    )


def decompose_query_node(state: AgentState, cfg: Any) -> PartialState:
    mappings = state.get("entity_mappings", {})
    decomposition: Dict[str, Any] = {
        "sub_queries": [],
        "execution_strategy": "single",
        "reasoning": "Default single-query strategy",
    }

    table_counts: Dict[str, int] = defaultdict(int)
    for mapping in mappings.values():
        final_mapping = mapping.get("final_mapping") or {}
        table = final_mapping.get("table")
        if table:
            table_counts[table] += 2  # prioritize final mapping

        for match in mapping.get("pattern_match", {}).get("matches", []):
            match_table = match.get("table")
            if match_table:
                table_counts[match_table] += 1

    query_text = state.get("user_input", "").lower()
    for table in list(table_counts.keys()):
        normalized = table.lower().replace("_data", "").replace("_table", "")
        if normalized and normalized in query_text:
            table_counts[table] += 3

    sorted_tables = sorted(table_counts.items(), key=lambda x: x[1], reverse=True)
    if not sorted_tables:
        reasoning = "No mapped entities, returning empty decomposition"
        return _make_node_result(
            state,
            node_name="decompose_query",
            control="construct_queries",
            reasoning=reasoning,
            state_updates={"query_decomposition": decomposition},
        )

    primary_table = sorted_tables[0][0]
    sub_queries: List[Dict[str, Any]] = []
    sub_queries.append(
        {
            "id": "sq_primary",
            "description": f"Analyze {primary_table} for matched entities",
            "table": primary_table,
        }
    )

    for table, count in sorted_tables[1:]:
        sub_queries.append(
            {
                "id": f"sq_{table}",
                "description": f"Collect supporting data from {table}",
                "table": table,
            }
        )

    execution_strategy = "single" if len(sub_queries) == 1 else "sequential"
    decomposition = {
        "sub_queries": sub_queries,
        "execution_strategy": execution_strategy,
        "reasoning": "Generated based on entity-to-table mappings",
    }

    reasoning = f"Decomposed request into {len(sub_queries)} query blocks"
    return _make_node_result(
        state,
        node_name="decompose_query",
        control="construct_queries",
        reasoning=reasoning,
        state_updates={"query_decomposition": decomposition},
    )


def construct_queries_node(state: AgentState, cfg: Any) -> PartialState:
    sql_queries: List[Dict[str, Any]] = []
    text_values = state.get("text_column_values", {})

    for sub_query in state.get("query_decomposition", {}).get("sub_queries", []):
        table = sub_query.get("table")
        if not table:
            continue

        filters: List[str] = []
        for entity_text, mapping in state.get("entity_mappings", {}).items():
            final_mapping = mapping.get("final_mapping") or {}
            target_column = None
            if final_mapping.get("table") == table:
                target_column = final_mapping.get("column")
            else:
                for match in mapping.get("pattern_match", {}).get("matches", []):
                    if match.get("table") == table:
                        target_column = match.get("column")
                        break

            if target_column:
                filters.append(
                    f"LOWER({target_column}) LIKE '%{entity_text.lower()}%'"
                )

        where_clause = " AND ".join(filters) if filters else "1=1"
        limit = cfg.get_nested("query", "default_limit", default=200)
        sql = f"SELECT * FROM {table} WHERE {where_clause} LIMIT {limit};"

        sql_queries.append(
            {
                "id": sub_query.get("id"),
                "table": table,
                "sql": sql,
                "description": sub_query.get("description"),
            }
        )

    reasoning = f"Constructed {len(sql_queries)} SQL templates"
    return _make_node_result(
        state,
        node_name="construct_queries",
        control="validate_queries",
        reasoning=reasoning,
        state_updates={"sql_queries": sql_queries},
    )


def validate_queries_node(state: AgentState, cfg: Any) -> PartialState:
    invalid = [query for query in state.get("sql_queries", []) if not query.get("sql")]
    if invalid:
        reasoning = "Validation failed: missing SQL text"
        return _make_node_result(
            state,
            node_name="validate_queries",
            control="end",
            reasoning=reasoning,
            state_updates={"sql_queries": []},
        )

    reasoning = "Queries validated"
    return _make_node_result(
        state,
        node_name="validate_queries",
        control="tool_decision",
        reasoning=reasoning,
    )


def tool_decision_node(state: AgentState, cfg: Any) -> PartialState:
    loop_count = int(state.get("tool_loop_count", 0))
    max_loops = cfg.get_nested("tool_loop", "max_turns", default=5)
    pending = state.get("tool_requests", [])

    if loop_count >= max_loops or not pending:
        reasoning = "No additional tool calls needed"
        return _make_node_result(
            state,
            node_name="tool_decision",
            control="generate_reasoning",
            reasoning=reasoning,
            state_updates={"tool_loop_count": loop_count},
        )

    reasoning = "Dispatching additional tool call"
    return _make_node_result(
        state,
        node_name="tool_decision",
        control="select_tool",
        reasoning=reasoning,
        state_updates={"tool_loop_count": loop_count + 1},
    )


def select_tool_node(state: AgentState, cfg: Any) -> PartialState:
    pending = state.get("tool_requests", [])
    if not pending:
        reasoning = "No tool requests in queue"
        return _make_node_result(
            state,
            node_name="select_tool",
            control="generate_reasoning",
            reasoning=reasoning,
        )

    request = pending.pop(0)
    state.setdefault("tool_responses", []).append({"request": request, "status": "skipped"})
    reasoning = "Tool loop placeholder executed"
    return _make_node_result(
        state,
        node_name="select_tool",
        control="tool_decision",
        reasoning=reasoning,
        state_updates={"tool_requests": pending},
    )


def generate_reasoning_node(state: AgentState, cfg: Any) -> PartialState:
    reasoning_doc = {
        "intent": state.get("intent_snapshot", {}),
        "entities": state.get("extracted_entities", []),
        "entity_mappings": state.get("entity_mappings", {}),
        "entity_search": state.get("entity_search_results", {}),
        "query_plan": state.get("query_decomposition", {}),
        "sql_queries": state.get("sql_queries", []),
    }
    reasoning_text = json.dumps(reasoning_doc, indent=2)
    return _make_node_result(
        state,
        node_name="generate_reasoning",
        control="synthesize",
        reasoning="Compiled analytical reasoning document",
        state_updates={"analytical_reasoning": reasoning_doc, "reasoning_text": reasoning_text},
    )


def synthesize_response_node(state: AgentState, cfg: Any) -> PartialState:
    final_output = {
        "response": "Deep research plan ready",
        "analytical_reasoning": state.get("analytical_reasoning", {}),
        "sql_queries": state.get("sql_queries", []),
    }
    return _make_node_result(
        state,
        node_name="synthesize",
        control="end",
        reasoning="Finalized deep research output",
        state_updates={"final_output": final_output},
    )


# Backward compatibility alias
def end_node(state: AgentState, cfg: Any) -> PartialState:
    return synthesize_response_node(state, cfg)
