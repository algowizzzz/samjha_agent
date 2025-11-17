"""Search for entity occurrences across structured data tables."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import duckdb

from tools.base_mcp_tool import BaseMCPTool


class SearchEntityInDataTool(BaseMCPTool):
    """
    Executes DuckDB search queries across provided tables/columns to find
    occurrences of an entity string (Ctrl+F style search).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string"},
                "tables": {
                    "type": "array",
                    "description": "Table metadata with paths and text columns.",
                },
                "case_sensitive": {"type": "boolean"},
                "max_results_per_column": {"type": "integer"},
            },
            "required": ["entity_value", "tables"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_value": {"type": "string"},
                "total_matches": {"type": "integer"},
                "matches": {"type": "array"},
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_arguments(arguments)

        entity_value = arguments["entity_value"]
        tables = [
            table for table in arguments.get("tables", []) if isinstance(table, dict)
        ]
        case_sensitive = bool(arguments.get("case_sensitive", False))
        max_results_per_column = max(1, int(arguments.get("max_results_per_column", 10)))

        if not tables:
            raise ValueError("No tables supplied for entity search.")

        matches: List[Dict[str, Any]] = []

        for table_meta in tables:
            try:
                table_matches = self._search_table(
                    entity_value=entity_value,
                    table_meta=table_meta,
                    case_sensitive=case_sensitive,
                    max_results=max_results_per_column,
                )
                matches.extend(table_matches)
            except Exception as exc:
                matches.append(
                    {
                        "table": table_meta.get("name"),
                        "error": str(exc),
                    }
                )

        total_found = sum(match.get("match_count", 0) for match in matches)
        return {
            "entity_value": entity_value,
            "total_matches": total_found,
            "matches": matches,
        }

    # Internal helpers -------------------------------------------------
    def _search_table(
        self,
        entity_value: str,
        table_meta: Dict[str, Any],
        case_sensitive: bool,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        table_name = table_meta.get("name") or table_meta.get("table")
        source_path = table_meta.get("path")
        source_format = (table_meta.get("format") or "").lower()
        text_columns = table_meta.get("text_columns") or []
        if not table_name or not source_path or not text_columns:
            raise ValueError("Table metadata must include name, path, and text_columns.")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Table source not found: {source_path}")

        conn = duckdb.connect(database=":memory:")
        view_name = f"tbl_{abs(hash(source_path))}"
        self._register_source(conn, view_name, source_path, source_format)

        matches: List[Dict[str, Any]] = []
        for column in text_columns:
            try:
                column_matches = self._search_column(
                    conn,
                    view_name,
                    table_name,
                    column,
                    entity_value,
                    case_sensitive,
                    max_results,
                )
                if column_matches:
                    matches.append(column_matches)
            except Exception as exc:
                matches.append(
                    {
                        "table": table_name,
                        "column": column,
                        "error": str(exc),
                    }
                )

        conn.close()
        return matches

    @staticmethod
    def _register_source(
        conn: duckdb.DuckDBPyConnection,
        view_name: str,
        path: str,
        fmt: str,
    ) -> None:
        if fmt == "parquet" or path.endswith(".parquet"):
            conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
        else:
            # Default to CSV auto-detection
            conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_csv_auto('{path}')"
            )

    @staticmethod
    def _search_column(
        conn: duckdb.DuckDBPyConnection,
        view_name: str,
        table_name: str,
        column: str,
        entity_value: str,
        case_sensitive: bool,
        max_results: int,
    ) -> Dict[str, Any]:
        placeholder = "?"
        if case_sensitive:
            predicate = f"CAST({column} AS VARCHAR) LIKE {placeholder}"
            search_value = f"%{entity_value}%"
        else:
            predicate = f"lower(CAST({column} AS VARCHAR)) LIKE {placeholder}"
            search_value = f"%{entity_value.lower()}%"

        count_query = (
            f"SELECT COUNT(*) FROM {view_name} WHERE {predicate}"
        )
        match_count = conn.execute(count_query, [search_value]).fetchone()[0]
        if match_count == 0:
            return {}

        sample_query = (
            f"SELECT DISTINCT {column} FROM {view_name} "
            f"WHERE {predicate} "
            f"LIMIT {max_results}"
        )
        rows = conn.execute(sample_query, [search_value]).fetchall()
        sample_values = [row[0] for row in rows if row and row[0] is not None]

        return {
            "table": table_name,
            "column": column,
            "match_count": int(match_count),
            "sample_values": sample_values,
            "case_sensitive": case_sensitive,
        }
