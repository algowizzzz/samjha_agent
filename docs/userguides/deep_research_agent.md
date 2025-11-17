# Deep Research Agent

The Deep Research Agent discovers local structured data sources (CSV/Parquet), maps natural-language entities to columns, and produces ready-to-run SQL plans plus analytical reasoning. It is designed to run ahead of the DuckDB-backed Parquet agent when business knowledge and schemas are not yet known.

## Workflow summary

1. **Discovery** – scans the configured `data/search_roots` for CSV/Parquet files and records table metadata.
2. **Schema extraction** – infers column names and types and captures sample text values for entity mapping.
3. **Entity understanding** – extracts entity candidates, captures the user’s intent snapshot, and generates column embeddings.
4. **Mapping & search** – runs pattern, vector, and LLM-based mapping plus a DuckDB-powered Ctrl+F search across all text columns.
5. **Query planning** – decomposes complex questions into sub-queries, generates SQL templates, validates them, and records the execution plan.
6. **Reasoning output** – produces an analytical reasoning document and returns the query plan for downstream agents to execute.

## Configuration

Configuration lives in `external/config/agent/deep_research_agent.json`. Key settings:

- `data.search_roots` – directories that will be scanned for CSV/Parquet files.
- `data.extensions` – file extensions to include.
- `data.text_value_limit` – distinct values to collect per text column.
- `query.default_limit` – default LIMIT applied to generated SQL.
- `tool_loop.max_turns` – maximum optional tool calls in the tail-loop.

You can supply a custom config path when instantiating `DeepResearchAgent(config_path=...)` (tests use this to point at temporary datasets).

## MCP tools added

| Tool | Description |
| --- | --- |
| `pattern_match_entity_column` | Heuristic entity→column mapping via lexical matches |
| `vector_search_entity_column` | Embedding similarity search against the generated index |
| `llm_map_entity_column` | LLM-assisted disambiguation with heuristic fallback |
| `create_column_embeddings` | Builds the embeddings index for all discovered columns |
| `search_entity_in_data` | Runs DuckDB `LIKE` searches across all text columns (Ctrl+F experience) |

## Testing

Unit tests live in `test/test_deep_research_tools.py` and cover each MCP tool. Execute them with:

```bash
./venv/bin/pytest test/test_deep_research_tools.py
```

End-to-end workflow tests live in `test/test_deep_research_agent.py`. They spin up a temporary dataset and verify five representative scenarios:

1. Simple exposure lookup
2. Multi-table decomposition
3. Financial analysis (parallel tables)
4. Ambiguous entity reasoning
5. Entity-not-found handling

Run them with:

```bash
./venv/bin/pytest test/test_deep_research_agent.py
```

Both suites patch the embedding model to avoid heavy downloads, so they run quickly in CI.

## Notes

- `requirements.txt` now includes `chromadb` and `sentence-transformers` for embedding support.
- The agent is accessible via the new MCP tool config `config/tools/deep_research_agent.json`.
- The agent does not execute SQL; it prepares the SQL plus reasoning for a downstream Parquet executor.
