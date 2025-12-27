# Backend Technical Documentation: MCP Server & Parquet Agent

**Version:** 1.0  
**Date:** December 2024  
**Status:** Development  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Reference](#2-component-reference)
3. [Data Flow](#3-data-flow)
4. [API Reference](#4-api-reference)
5. [WebSocket Events](#5-websocket-events)
6. [Response Schemas](#6-response-schemas)
7. [Storage & Persistence](#7-storage--persistence)
8. [Known Issues & Required Fixes](#8-known-issues--required-fixes)
9. [Task Backlog](#9-task-backlog)
10. [Testing Guide](#10-testing-guide)

---

## 1. Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flask HTTP Server                         │
│  POST /api/mcp  ───►  MCPHandler (JSON-RPC 2.0)                 │
│  /agent/chat    ───►  HTML UI Page                              │
│  WebSocket      ───►  SocketIO (agent:query, agent:kill)        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ToolsRegistry                              │
│  - Singleton pattern                                            │
│  - Loads tools from config/tools/*.json                         │
│  - Hot-reload on file changes (5s polling)                      │
│  - 50+ tools available                                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ parquet_agent tool
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ParquetQueryAgent                           │
│                   (Decider → Executor Loop)                      │
│                      Max 3 retry attempts                        │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| Web Framework | Flask + Flask-SocketIO |
| LLM Provider | Anthropic Claude (claude-3-haiku) |
| Database Engine | DuckDB (in-memory) |
| State Storage | JSON files |
| Protocol | JSON-RPC 2.0 (MCP) |

---

## 2. Component Reference

### Core Files

| File | Purpose | Lines |
|------|---------|-------|
| `core/mcp_handler.py` | JSON-RPC 2.0 protocol handler | ~300 |
| `tools/tools_registry.py` | Tool loading & management (singleton) | ~390 |
| `tools/base_mcp_tool.py` | Base class for all tools | ~220 |

### Agent Files

| File | Purpose | Lines |
|------|---------|-------|
| `external/agent/parquet_agent.py` | Main controller loop | ~340 |
| `external/agent/decider.py` | LLM decision maker (ASK_USER/EXECUTE/BLOCK) | ~240 |
| `external/agent/executor_graph.py` | LangGraph executor pipeline | ~160 |
| `external/agent/executor_nodes.py` | 6 executor nodes (investigation → outcome) | ~350 |
| `external/agent/graph_nodes.py` | Full graph node implementations | ~1700 |
| `external/agent/streaming_agent.py` | WebSocket streaming wrapper | ~900 |
| `external/agent/state_manager.py` | JSON-based state persistence | ~100 |
| `external/agent/session_manager.py` | Active session & cancellation tracking | ~70 |
| `external/agent/schemas.py` | AgentState TypedDict definitions | ~100 |
| `external/agent/state_types.py` | ControllerState & ExecutorState types | ~35 |

### Route Files

| File | Purpose |
|------|---------|
| `routes/api_routes.py` | MCP endpoint, tool execution |
| `external/routes/agent_routes.py` | Agent chat UI, config management |
| `external/routes/agent_socketio_handlers.py` | WebSocket event handlers |

### Platform Abstractions

| File | Purpose |
|------|---------|
| `external/platform/llm/client.py` | Unified LLM client (Anthropic) |
| `external/platform/storage/state_manager.py` | State persistence |
| `external/platform/storage/session_manager.py` | Session tracking |
| `external/platform/agent/streaming.py` | Streaming agent |

### Tool Implementations (Parquet Agent)

| Tool Name | File | Purpose |
|-----------|------|---------|
| `list_dir` | `external/tools/parquet_agent/list_dir.py` | List files in data directory |
| `inspect_table` | `external/tools/parquet_agent/inspect_table.py` | Get table schema |
| `preview_rows` | `external/tools/parquet_agent/preview_rows.py` | Sample table data |
| `search_glossary` | `external/tools/parquet_agent/search_glossary.py` | Search data dictionary |
| `nl_to_sql_planner` | `external/tools/parquet_agent/nl_to_sql_planner.py` | Generate SQL from NL |
| `sql_plan_updater` | `external/tools/parquet_agent/sql_plan_updater.py` | Patch SQL plan |
| `query_safety_validator` | `external/tools/parquet_agent/query_safety_validator.py` | Validate SQL safety |
| `execute_sql` | `external/tools/parquet_agent/execute_sql.py` | Run SQL on DuckDB |
| `query_result_evaluator` | `external/tools/parquet_agent/query_result_evaluator.py` | Evaluate results |

---

## 3. Data Flow

### Query Processing Flow

```
User Query
    │
    ▼
┌─────────────────┐
│    Decider      │◄── domain_md + conversation_history
│  (1 LLM call)   │
└────────┬────────┘
         │
    ┌────┴────────────────┐
    ▼                     ▼
ASK_USER/BLOCK        EXECUTE
    │                     │
    ▼                     ▼
Return to UI    ┌─────────────────┐
                │ Executor Graph  │
                │  (6 nodes)      │
                └────────┬────────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
                SUCCESS     ERROR
                    │         │
                    └────┬────┘
                         ▼
                Return to Controller
                (retry up to 3x on ERROR)
```

### Executor Pipeline (6 Nodes)

```
1. InvestigationNode
   ├── Runs tool calls from investigation_plan
   └── Updates query_spec with discovered info

2. SQLGenerationNode
   ├── Gate: spec_ready_for_sql() check
   └── Uses nl_to_sql_planner tool

3. SafetyValidationNode
   └── Uses query_safety_validator tool

4. ExecutionNode
   └── Uses execute_sql tool

5. EvaluationNode
   └── Uses query_result_evaluator tool

6. OutcomeNode
   └── Builds SUCCESS or ERROR report
```

---

## 4. API Reference

### Existing Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/mcp` | POST | Bearer | MCP JSON-RPC 2.0 endpoint |
| `/api/tools/execute` | POST | Bearer | Direct tool execution |
| `/api/tools/list` | GET | Session | List available tools |
| `/api/tools/<name>/schema` | GET | Session | Get tool schema |
| `/agent/chat` | GET | Session | Chat UI page |
| `/api/agent/welcome-message` | GET | Session | Get welcome JSON |
| `/api/agent/config-files` | GET | Session | List config files |
| `/api/agent/config-upload` | POST | Session | Upload config file |
| `/api/agent/data-analysis-agents` | GET/POST | Session | Agent presets CRUD |
| `/api/agent/data-analysis-agents/<id>` | GET | Session | Get specific agent |

### Missing Endpoints (TO BE IMPLEMENTED)

| Endpoint | Method | Description | Priority |
|----------|--------|-------------|----------|
| `/api/agent/history/<session_id>` | GET | Get conversation history | P1 |
| `/api/agent/sessions` | GET | List user's sessions | P1 |
| `/api/agent/query` | POST | REST fallback for query | P2 |
| `/api/agent/history/<session_id>` | DELETE | Clear session history | P2 |

---

## 5. WebSocket Events

### Client → Server

| Event | Payload | Description |
|-------|---------|-------------|
| `agent:query` | `{query, session_id, token, data_dict_file?, agent_prompts_file?, user_clarification?}` | Send query |
| `agent:kill` | `{session_id, token}` | Cancel running query |

### Server → Client

| Event | Payload | Description |
|-------|---------|-------------|
| `agent:node_start` | `{node, session_id}` | Node execution started |
| `agent:llm_chunk` | `{chunk, node, session_id}` | Streaming LLM text |
| `agent:node_complete` | `{node, session_id}` | Node finished |
| `agent:node_data` | `{node, session_id, data}` | Node output data |
| `agent:complete` | `{result, session_id}` | Query completed |
| `agent:waiting_for_clarification` | `{clarify_prompt, session_id}` | Needs user input |
| `agent:cancelled` | `{message, session_id}` | Query was cancelled |
| `agent:killed` | `{message, session_id}` | Kill confirmed |
| `agent:error` | `{error, session_id}` | Error occurred |
| `agent:kill_error` | `{error, session_id}` | Kill failed |

---

## 6. Response Schemas

### ParquetQueryAgent.run_query() Response

#### SUCCESS
```json
{
  "session_id": "uuid",
  "user_id": "user123",
  "control": "end",
  "final_output": {
    "response": "Query executed successfully. Returned 50 rows with 5 columns.",
    "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
    "result_summary": "Returned 50 rows with 5 columns"
  },
  "plan": {
    "sql": "SELECT region, SUM(revenue) FROM sales GROUP BY region"
  }
}
```

#### ASK_USER (Clarification Needed)
```json
{
  "session_id": "uuid",
  "user_id": "user123",
  "control": "wait_for_user",
  "clarification": {
    "questions": ["Which region do you want to filter by?"],
    "reasoning": ["Multiple regions exist in the data"],
    "prompt": "Please specify the region to continue"
  },
  "query_spec": { ... },
  "query_spec_status": { ... }
}
```

#### BLOCK (Refused)
```json
{
  "session_id": "uuid",
  "user_id": "user123",
  "control": "end",
  "final_output": {
    "response": "❌ Query blocked: Query violates safety policy"
  }
}
```

#### ERROR
```json
{
  "session_id": "uuid",
  "user_id": "user123",
  "control": "end",
  "final_output": {
    "response": "❌ Error: Max attempts reached"
  },
  "error": {
    "status": "ERROR",
    "reason": "Max attempts reached",
    "attempt_count": 3,
    "last_executor_report": { ... }
  }
}
```

### AgentState Fields (Full State Object)

```typescript
interface AgentState {
  // Inputs
  user_input: string;
  user_id?: string;
  session_id?: string;
  timestamp?: string;
  
  // Enrichment
  docs_meta: Array<{table: string, columns: string[], row_count: number}>;
  table_schema: Record<string, any>;
  parquet_location: string;
  
  // Conversation
  conversation_history?: Array<{query: string, response: string}>;
  conversation_history_raw?: Array<{query, sql, raw_table, response, prompt_monitor}>;
  
  // Control
  control?: ControlSignal; // "invoke" | "check_structure" | "clarify" | "generate_sql" | "execute_sql" | "end" | ...
  last_node?: string;
  node_reasoning?: string;
  
  // Node outputs
  is_structured?: boolean;
  is_ambiguous?: boolean;
  clarification_questions?: string[];
  plan?: {sql: string, target_table: string, explanation: string};
  plan_quality?: "high" | "medium" | "low" | "error";
  execution_result?: {columns: string[], rows: any[], row_count: number, query: string};
  raw_table?: {columns: string[], rows: any[], row_count: number};
  
  // Final
  final_output?: {raw_table: any, response: string, prompt_monitor: string};
  
  // Metrics
  metrics?: {node_timings_ms: Record<string, number>, total_ms: number, clarify_turns: number};
}
```

---

## 7. Storage & Persistence

### State File Location
```
data/agent_state/session_{session_id}.json
```

### State File Schema
```json
{
  "session_id": "uuid-here",
  "user_id": "user123",
  "created_at": "2024-12-26T10:00:00.000Z",
  "updated_at": "2024-12-26T10:05:00.000Z",
  "state": {
    "user_input": "...",
    "conversation_history_raw": [...],
    "final_output": {...},
    ...
  }
}
```

### Config File Locations

| Type | Directory |
|------|-----------|
| Data Dictionary | `external/config/data_dictionary/` |
| Agent Prompts | `external/config/agent/` |
| Welcome Tips | `external/config/agent_welcome/` |
| Tool Configs | `config/tools/` + `external/config/tools/` |
| Agent Presets | `external/config/data_analysis_agents/` |
| Domain Instructions | `domain_instructions/` |

---

## 8. Known Issues & Required Fixes

### 8.1 Hardcoded Debug Logging (CRITICAL)

**Files Affected:**
- `external/agent/executor_nodes.py` (9 instances)
- `external/tools/parquet_agent/query_result_evaluator.py` (4 instances)

**Problem:** Debug logs write to hardcoded absolute path:
```python
with open('/Users/saadahmed/Desktop/samjha_agent-1/.cursor/debug.log', 'a') as f:
```

**Lines to Remove:**

| File | Lines |
|------|-------|
| `executor_nodes.py` | 262, 268, 279, 291, 304, 315, 324, 342, 351 |
| `query_result_evaluator.py` | 149, 198, 205, 212 |

**Fix:** Remove all debug blocks or replace with proper logging:
```python
logger.debug(f"Evaluation node entry: halt={halt_execution}")
```

---

### 8.2 Duplicate Files (100% Identical)

| Original | Duplicate | Action |
|----------|-----------|--------|
| `external/platform/storage/session_manager.py` | `external/agent/session_manager.py` | Delete agent version |
| `external/platform/storage/state_manager.py` | `external/agent/state_manager.py` | Delete agent version |
| `external/platform/agent/streaming.py` | `external/agent/streaming_agent.py` | Delete agent version |

**After Deletion:** Update all imports to use `external.platform.*`

---

### 8.3 Import Inconsistency

**File:** `external/routes/agent_socketio_handlers.py`

| Line | Current Import | Should Be |
|------|----------------|-----------|
| 74 | `from external.platform.storage import AgentSessionManager` | ✅ Correct |
| 141 | `from external.agent.session_manager import AgentSessionManager` | ❌ Use platform |

---

### 8.4 Tool Config Location

**Problem:** `parquet_agent.json` is in `external/config/tools/` but ToolsRegistry loads from `config/tools/` by default.

**Fix:** Copy `external/config/tools/parquet_agent.json` → `config/tools/parquet_agent.json`

---

### 8.5 Missing Schema Files

**Verify these exist:**
- `external/schemas/decider_output.schema.json`
- `external/schemas/executor_report.schema.json`

---

## 9. Task Backlog

### P0 - Critical (Before Deploy)

| # | Task | File | Effort | Status |
|---|------|------|--------|--------|
| 1 | Remove hardcoded debug logging | `executor_nodes.py`, `query_result_evaluator.py` | 15m | ⬜ |
| 2 | Standardize imports to `external.platform.*` | `agent_socketio_handlers.py:141` | 5m | ⬜ |
| 3 | Verify schema JSON files exist | `external/schemas/` | 10m | ⬜ |
| 4 | Copy `parquet_agent.json` to `config/tools/` | File operation | 2m | ⬜ |

### P1 - High (Frontend Needs)

| # | Task | File | Effort | Status |
|---|------|------|--------|--------|
| 5 | Add `GET /api/agent/history/<session_id>` | `agent_routes.py` | 30m | ⬜ |
| 6 | Add `GET /api/agent/sessions` | `agent_routes.py` | 30m | ⬜ |

### P2 - Medium (Nice to Have)

| # | Task | File | Effort | Status |
|---|------|------|--------|--------|
| 7 | Add `POST /api/agent/query` (REST fallback) | `agent_routes.py` | 1h | ⬜ |
| 8 | Add `DELETE /api/agent/history/<session_id>` | `agent_routes.py` | 20m | ⬜ |
| 9 | Delete duplicate files in `external/agent/` | File deletion | 10m | ⬜ |
| 10 | Update all imports after deletion | Various | 30m | ⬜ |

### P3 - Low (Cleanup)

| # | Task | File | Effort | Status |
|---|------|------|--------|--------|
| 11 | Replace print() with logging | Various | 30m | ⬜ |
| 12 | Add input validation for API endpoints | `agent_routes.py` | 45m | ⬜ |
| 13 | Add rate limiting | `agent_socketio_handlers.py` | 1h | ⬜ |

---

## 10. Testing Guide

### Environment Setup

```bash
# Required environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-3-haiku-20240307"  # Optional, default
export ANTHROPIC_TEMPERATURE="0.2"  # Optional, default
export ANTHROPIC_MAX_TOKENS="4096"  # Optional, default
```

### Start Server

```bash
python run_server.py
# Server runs on http://localhost:8000
```

### Test MCP Endpoint

```bash
# Initialize
curl -X POST http://localhost:8000/api/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {"clientInfo": {"name": "test"}}}'

# List tools
curl -X POST http://localhost:8000/api/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 2, "params": {}}'
```

### Test Tool Execution

```bash
curl -X POST http://localhost:8000/api/tools/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "tool": "parquet_agent",
    "arguments": {
      "query": "Show me revenue by region"
    }
  }'
```

### Test WebSocket (using wscat)

```bash
npm install -g wscat
wscat -c ws://localhost:8000/socket.io/?EIO=4&transport=websocket

# Send authentication
42["authenticate",{"token":"<your-token>"}]

# Send query
42["agent:query",{"query":"revenue by region","session_id":"test-123","token":"<token>"}]

# Kill query
42["agent:kill",{"session_id":"test-123","token":"<token>"}]
```

### Verify State Persistence

```bash
# After running a query, check the state file
cat data/agent_state/session_<session_id>.json | jq .
```

---

## Appendix A: Decider Output Schema

The Decider produces a structured JSON output with these key fields:

```json
{
  "action": "EXECUTE | ASK_USER | BLOCK",
  "query_type": "NEW_QUERY | FOLLOW_UP | USER_ANSWER",
  "query_type_signals": ["signal1", "signal2"],
  "domain": "ecomm",
  "intent": "aggregation",
  
  "query_spec": {
    "business_question": "Revenue by region",
    "output_shape": {"type": "table", "columns": ["region", "revenue"]},
    "start_table": {"name": "sales_data", "path": "..."},
    "grain": "one row per region",
    "time": {"column": "order_date", "rule": "last_n_days", "n_days": 30},
    "metrics": [{"name": "revenue", "definition": "SUM(quantity * price)"}],
    "dimensions": ["region"],
    "filters": [],
    "joins": [],
    "aggregation_plan": "GROUP BY region",
    "validation_checks": [],
    "performance_guardrails": []
  },
  
  "query_spec_status": {
    "business_question": {"status": "verified", "source": "user", "blocks_execution": false},
    "grain": {"status": "inferred", "source": "domain_md", "blocks_execution": false}
  },
  
  "investigation_plan": [
    {"step": 1, "tool": "inspect_table", "args": {...}, "fills_gap": "grain"}
  ],
  
  "ask_user": {
    "question": "",
    "why_non_defaultable": "",
    "what_answer_unblocks": ""
  },
  
  "block_reason": ""
}
```

---

## Appendix B: Domain Configuration

Domain configs are stored in `domain_instructions/*.md` and define:

- Domain identity (key, description)
- Time semantics (default column, rule)
- Core entities (tables, grain hints)
- Dimensions dictionary
- Metrics dictionary
- Join conventions
- Safety defaults

See `domain_instructions/ecomm_domain.md` for reference.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 2024 | AI Assistant | Initial documentation |


