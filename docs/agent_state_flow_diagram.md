# Agent State Flow Diagram

## Complete Process Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT STATE (TypedDict)                            │
│  user_input, user_id, session_id, table_schema, docs_meta, parquet_location│
│  plan, plan_quality, plan_explain, clarification_questions                  │
│  execution_result, execution_stats, satisfaction, evaluator_notes           │
│  final_output, conversation_history, logs, metrics, control, last_node        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INVOKE NODE                                      │
│  Reads: user_input, user_id, session_id (from API)                         │
│  Writes: table_schema, docs_meta, parquet_location, conversation_history    │
│  Sets: control="plan", last_node="invoke"                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PLANNER NODE                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  NLToSQLPlannerTool.execute()                                    │     │
│  │                                                                   │     │
│  │  INPUTS FROM STATE:                                              │     │
│  │  ├─ query: state["user_input"]                                   │     │
│  │  ├─ table_schema: state.get("table_schema", {})                  │     │
│  │  ├─ docs_meta: state.get("docs_meta", [])                         │     │
│  │  └─ previous_clarifications: []                                  │     │
│  │                                                                   │     │
│  │  CONFIG INPUTS:                                                   │     │
│  │  ├─ preview_rows: cfg["duckdb"]["preview_rows"]                  │     │
│  │  ├─ data_directory: state["parquet_location"]                     │     │
│  │  └─ use_llm: True                                                │     │
│  │                                                                   │     │
│  │  OUTPUTS → STATE:                                                 │     │
│  │  ├─ plan: {sql, type, target_table}                              │     │
│  │  ├─ plan_quality: "high"|"medium"|"low"                          │     │
│  │  ├─ plan_explain: "explanation string"                           │     │
│  │  └─ clarification_questions: ["question1", "question2"]        │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  Sets: control="execute"|"clarify", last_node="planner"                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
          plan_quality="high"            plan_quality="low"|"medium"
                    │                               │
                    ▼                               ▼
┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│      EXECUTE NODE                 │  │      CLARIFY NODE                  │
│                                   │  │                                   │
│  ┌─────────────────────────────┐ │  │  Reads: clarification_questions    │
│  │ QuerySafetyValidatorTool    │ │  │  Writes: clarify_prompt,           │
│  │                             │ │  │         clarify_questions,         │
│  │ INPUTS FROM STATE:          │ │  │         clarify_reasoning          │
│  │ ├─ query: plan.sql         │ │  │  Sets: control="wait_for_user"    │
│  │ ├─ enforce_limit: True     │ │  │         metrics["clarify_turns"]++ │
│  │ └─ default_limit: 100       │ │  │                                   │
│  │                             │ │  │  ┌─────────────────────────────┐ │
│  │ OUTPUTS → STATE:            │ │  │  │ User provides clarification │ │
│  │ ├─ is_safe: boolean         │ │  │  │ → user_clarification        │ │
│  │ ├─ sanitized_query: string  │ │  │  └─────────────────────────────┘ │
│  │ └─ limit_enforced: boolean  │ │  │                                   │
│  └─────────────────────────────┘ │  │  ┌─────────────────────────────┐ │
│                                   │  │  │      REPLAN NODE            │ │
│  ┌─────────────────────────────┐ │  │  │                             │ │
│  │ DuckDB Direct Execution     │ │  │  │  ┌───────────────────────┐ │ │
│  │                             │ │  │  │  │ NLToSQLPlannerTool    │ │ │
│  │ INPUTS FROM STATE:          │ │  │  │  │                       │ │ │
│  │ ├─ parquet_location         │ │  │  │  │ INPUTS:              │ │ │
│  │ └─ sanitized_query          │ │  │  │  │ ├─ query: combined   │ │ │
│  │                             │ │  │  │  │ │  (user_input +      │ │ │
│  │ OUTPUTS → STATE:            │ │  │  │  │ │   user_clarification)│ │ │
│  │ ├─ execution_result: {      │ │  │  │  │ ├─ table_schema       │ │ │
│  │ │   columns: [...],          │ │  │  │  │ ├─ docs_meta          │ │ │
│  │ │   rows: [...],             │ │  │  │  │ └─ previous_clarif.  │ │ │
│  │ │   row_count: N,            │ │  │  │  │                       │ │ │
│  │ │   query: "SELECT..."      │ │  │  │  │ OUTPUTS:              │ │ │
│  │ │ }                          │ │  │  │  │ ├─ plan (updated)     │ │ │
│  │ ├─ execution_stats: {        │ │  │  │  │ ├─ plan_quality       │ │ │
│  │ │   execution_time_ms: X,    │ │  │  │  │ └─ clarification_qs  │ │ │
│  │ │   error: null,             │ │  │  │  └───────────────────────┘ │ │
│  │ │   limited: false           │ │  │  │                             │ │
│  │ │ }                          │ │  │  │  Sets: control="execute"    │ │
│  │ └─ metrics.row_count: N      │ │  │  └─────────────────────────────┘ │
│  └─────────────────────────────┘ │  │                                   │
│                                   │  │         ┌──────────────────────┐  │
│  Sets: control="evaluate"         │  └────────► Back to EXECUTE NODE │  │
│         last_node="execute"        │            └──────────────────────┘  │
└───────────────────────────────────┘                                      │
                    │                                                       │
                    ▼                                                       │
┌───────────────────────────────────────────────────────────────────────────┐
│                         EVALUATE NODE                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  QueryResultEvaluatorTool.execute()                              │     │
│  │                                                                   │     │
│  │  INPUTS FROM STATE:                                              │     │
│  │  ├─ original_query: state["user_input"]                         │     │
│  │  ├─ execution_result: state.get("execution_result", {})          │     │
│  │  │    {columns, rows, row_count, query}                         │     │
│  │  ├─ execution_stats: state.get("execution_stats", {})           │     │
│  │  │    {execution_time_ms, error, limited}                        │     │
│  │  └─ table_schema: state.get("table_schema", {})                 │     │
│  │                                                                   │     │
│  │  CONFIG INPUTS:                                                   │     │
│  │  └─ use_llm: True                                                │     │
│  │                                                                   │     │
│  │  OUTPUTS → STATE:                                                │     │
│  │  ├─ satisfaction: "satisfied"|"needs_work"|"failed"            │     │
│  │  ├─ evaluator_notes: "assessment string"                        │     │
│  │  ├─ issues_detected: ["issue1", "issue2"]                      │     │
│  │  └─ suggested_improvements: ["suggestion1"]                   │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  Sets: control="end"|"replan", last_node="evaluate"                         │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
satisfaction="satisfied"  satisfaction="needs_work"
        │                       │
        ▼                       ▼
┌───────────────────┐  ┌───────────────────┐
│    END NODE       │  │  REPLAN NODE      │
│                   │  │  (loop back)      │
│  Reads:           │  └───────────────────┘
│  ├─ user_input    │         │
│  ├─ plan          │         │
│  ├─ execution_result│       │
│  ├─ execution_stats│        │
│  ├─ satisfaction  │        │
│  └─ evaluator_notes│        │
│                   │         │
│  Writes:          │         │
│  ├─ final_output: {│        │
│  │   response: "user-friendly summary",│
│  │   prompt_monitor: {full state summary}│
│  │ }              │         │
│  ├─ conversation_history: [│
│  │   {query, plan_sql, response, satisfaction, ...}│
│  │ ] (last 5 only)│         │
│  └─ metrics.total_ms│       │
│                   │         │
│  Sets: control="end"│        │
│         last_node="end"│     │
└───────────────────┘         │
                              │
                              └──► Back to PLANNER NODE
```

---

## Detailed Tool Interaction Matrix

### Tool 1: NLToSQLPlannerTool

```
┌─────────────────────────────────────────────────────────────┐
│ STATE INPUTS                    │  TOOL PROCESS              │
├─────────────────────────────────┼───────────────────────────┤
│ user_input: "sales by region"    │  ┌─────────────────────┐  │
│ table_schema: {                  │  │ 1. Parse query     │  │
│   "market_limits": {columns:[]} │  │ 2. Match to schema │  │
│ }                               │  │ 3. Generate SQL     │  │
│ docs_meta: [                    │  │ 4. Assess quality   │  │
│   {table: "market_limits", ...}, │  │ 5. Generate clarify │  │
│   {glossary: {...}}             │  │    questions        │  │
│ ]                               │  └─────────────────────┘  │
│ previous_clarifications: []     │                           │
└─────────────────────────────────┴───────────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STATE OUTPUTS                                               │
├─────────────────────────────────────────────────────────────┤
│ plan: {                                                     │
│   sql: "SELECT desk_id, SUM(remaining) FROM market_limits │
│         GROUP BY desk_id LIMIT 10",                        │
│   type: "sql_plan",                                         │
│   target_table: "market_limits"                            │
│ }                                                           │
│ plan_quality: "high"                                       │
│ plan_explain: "Query groups by desk and sums remaining"   │
│ clarification_questions: []                               │
└─────────────────────────────────────────────────────────────┘
```

**Called from**: `planner_node()` and `replan_node()`

**Inputs from State**:
- `query`: `state["user_input"]` (original user query)
- `table_schema`: `state.get("table_schema", {})` (table/column schema)
- `docs_meta`: `state.get("docs_meta", [])` (business context + glossary)
- `previous_clarifications`: `[]` (in planner) or `[user_clarification]` (in replan)

**Tool Config**:
- `preview_rows`: from `cfg["duckdb"]["preview_rows"]` (default: 100)
- `data_directory`: from `state["parquet_location"]`
- `use_llm`: `True`

**Outputs to State**:
- `plan`: dict with `sql`, `type`, `target_table`
- `plan_quality`: `"high"` | `"medium"` | `"low"`
- `plan_explain`: string explanation
- `clarification_questions`: list of strings (AI-generated)

---

### Tool 2: QuerySafetyValidatorTool

```
┌─────────────────────────────────────────────────────────────┐
│ STATE INPUTS                    │  TOOL PROCESS              │
├─────────────────────────────────┼───────────────────────────┤
│ plan.sql: "SELECT desk_id, ..." │  ┌─────────────────────┐  │
│                                  │  │ 1. Check for DDL   │  │
│                                  │  │    (CREATE, DROP)  │  │
│                                  │  │ 2. Check for DML   │  │
│                                  │  │    (UPDATE, DELETE)│  │
│                                  │  │ 3. Check LIMIT     │  │
│                                  │  │ 4. Add LIMIT if    │  │
│                                  │  │    missing         │  │
│                                  │  └─────────────────────┘  │
└─────────────────────────────────┴───────────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STATE OUTPUTS                                               │
├─────────────────────────────────────────────────────────────┤
│ execution_stats.limit_enforced: true                       │
│ (sanitized_query used for execution)                        │
└─────────────────────────────────────────────────────────────┘
```

**Called from**: `execute_node()`

**Inputs from State**:
- `query`: `(state.get("plan") or {}).get("sql", "SELECT 1")` (SQL from plan)
- `enforce_limit`: `True` (always)
- `default_limit`: `preview_rows` from config (default: 100)

**Tool Config**:
- `limits`: from `cfg.get("limits")` or `{"max_rows": 1000}`
- `safety`: from `cfg.get("safety")` or `{}`

**Outputs to State**:
- `is_safe`: boolean
- `reason`: string (if unsafe)
- `sanitized_query`: string (with LIMIT added if needed)
- `limit_enforced`: boolean

---

### Tool 3: QueryResultEvaluatorTool

```
┌─────────────────────────────────────────────────────────────┐
│ STATE INPUTS                    │  TOOL PROCESS              │
├─────────────────────────────────┼───────────────────────────┤
│ user_input: "sales by region"   │  ┌─────────────────────┐  │
│ execution_result: {             │  │ 1. Compare query    │  │
│   columns: ["desk_id", "sum"],  │  │    intent vs result │  │
│   rows: [{desk_id: "A", sum: 5}],│  │ 2. Check row count │  │
│   row_count: 1                  │  │ 3. Check columns    │  │
│ }                               │  │ 4. Assess quality   │  │
│ execution_stats: {              │  │ 5. Generate notes   │  │
│   execution_time_ms: 45,        │  │    and suggestions  │  │
│   error: null                   │  └─────────────────────┘  │
│ }                               │                           │
│ table_schema: {...}             │                           │
└─────────────────────────────────┴───────────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│ STATE OUTPUTS                                               │
├─────────────────────────────────────────────────────────────┤
│ satisfaction: "satisfied"                                   │
│ evaluator_notes: "Results match query intent. 1 row..."    │
│ issues_detected: []                                         │
│ suggested_improvements: []                                 │
└─────────────────────────────────────────────────────────────┘
```

**Called from**: `evaluate_node()`

**Inputs from State**:
- `original_query`: `state["user_input"]` (original user query)
- `execution_result`: `state.get("execution_result", {})` (full execution result with columns, rows, row_count, query)
- `execution_stats`: `state.get("execution_stats", {})` (execution metadata: execution_time_ms, error, limited)
- `table_schema`: `state.get("table_schema", {})` (table schema for context)

**Tool Config**:
- `use_llm`: `True` (always enabled)

**Outputs to State**:
- `satisfaction`: `"satisfied"` | `"needs_work"` | `"failed"`
- `evaluator_notes`: string assessment
- `issues_detected`: list of issues
- `suggested_improvements`: list of suggestions

---

## State Flow Summary

```
INVOKE
  │
  ├─► Reads: user_input, user_id, session_id
  └─► Writes: table_schema, docs_meta, parquet_location
      │
      ▼
PLANNER
  │
  ├─► Reads: user_input, table_schema, docs_meta
  ├─► Calls: NLToSQLPlannerTool
  └─► Writes: plan, plan_quality, plan_explain, clarification_questions
      │
      ├─► If plan_quality="high" ──► EXECUTE
      └─► If plan_quality="low"  ──► CLARIFY ──► REPLAN ──► EXECUTE
      │
      ▼
EXECUTE
  │
  ├─► Reads: plan.sql, parquet_location
  ├─► Calls: QuerySafetyValidatorTool ──► sanitized_query
  ├─► Executes: DuckDB direct query
  └─► Writes: execution_result, execution_stats
      │
      ▼
EVALUATE
  │
  ├─► Reads: user_input, execution_result, execution_stats, table_schema
  ├─► Calls: QueryResultEvaluatorTool
  └─► Writes: satisfaction, evaluator_notes
      │
      ├─► If satisfaction="satisfied" ──► END
      └─► If satisfaction="needs_work" ──► REPLAN ──► EXECUTE ──► EVALUATE
      │
      ▼
END
  │
  ├─► Reads: user_input, plan, execution_result, satisfaction, evaluator_notes
  └─► Writes: final_output, conversation_history (last 5), metrics.total_ms
```

---

## Tool Inputs Summary Table

| Tool | Node | State Inputs | Config Inputs |
|------|------|--------------|---------------|
| **NLToSQLPlannerTool** | Planner | `user_input`, `table_schema`, `docs_meta` | `preview_rows`, `parquet_location`, `use_llm` |
| **NLToSQLPlannerTool** | Replan | `user_input` + `user_clarification`, `table_schema`, `docs_meta` | Same as planner |
| **QuerySafetyValidatorTool** | Execute | `plan.sql` | `limits`, `safety` (from cfg) |
| **QueryResultEvaluatorTool** | Evaluate | `user_input`, `execution_result`, `execution_stats`, `table_schema` | `use_llm` |
| **DuckDB (direct)** | Execute | `parquet_location`, `plan` | None (uses sanitized SQL) |

---

## Key Patterns

1. **Read-only inputs**: Tools read from state but don't modify existing fields
2. **Append-only logs**: Each node appends to `logs` array
3. **State accumulation**: Each node adds new fields; previous fields remain unchanged
4. **Control flow**: `control` and `last_node` fields guide routing between nodes
5. **Tool isolation**: Tools receive inputs and return outputs; no direct state access

---

## State Evolution by Node

### 1. Invoke Node (Initialization)
**Adds**:
- `table_schema`: Discovered from CSV/Parquet files
- `docs_meta`: Loaded from `config/data_dictionary.json`
- `parquet_location`: From config
- `conversation_history`: Empty or loaded from previous session
- `control`: `"plan"`
- `last_node`: `"invoke"`
- `logs`: Initial log entry
- `metrics.start_time`: Timestamp

### 2. Planner Node
**Adds**:
- `plan`: SQL plan object
- `plan_quality`: `"high"` | `"medium"` | `"low"`
- `plan_explain`: Explanation string
- `clarification_questions`: AI-generated questions
- `control`: `"execute"` | `"clarify"`
- `last_node`: `"planner"`
- `metrics.plan_id`: Hash of query + plan

### 3. Clarify Node
**Adds**:
- `clarify_prompt`: Formatted prompt for user
- `clarify_questions`: Questions to ask
- `clarify_reasoning`: Reasons for asking
- `control`: `"wait_for_user"`
- `last_node`: `"clarify"`
- `metrics.clarify_turns`: Incremented counter

### 4. Replan Node
**Updates**:
- `plan`: New plan with user clarification
- `plan_quality`: Updated quality
- `plan_explain`: Updated explanation
- `clarification_questions`: New questions (if still needed)
- `control`: `"execute"` | `"clarify"`
- `last_node`: `"replan"`

### 5. Execute Node
**Adds**:
- `execution_result`: `{columns, rows, row_count, query}`
- `execution_stats`: `{execution_time_ms, error, limited}`
- `control`: `"evaluate"`
- `last_node`: `"execute"`
- `metrics.row_count`: Number of rows returned

### 6. Evaluate Node
**Adds**:
- `satisfaction`: `"satisfied"` | `"needs_work"` | `"failed"`
- `evaluator_notes`: Assessment string
- `control`: `"end"` | `"replan"`
- `last_node`: `"evaluate"`

### 7. End Node (Final)
**Adds**:
- `final_output`: `{response, prompt_monitor}`
- `conversation_history`: Updated with last 5 interactions
- `metrics.total_ms`: Total agent duration
- `control`: `"end"`
- `last_node`: `"end"`

---

## Notes

- All tools are **stateless** - they receive inputs from state and return outputs that update state
- State is **immutable** in the sense that tools don't modify existing fields, only add new ones
- The `control` field drives the flow between nodes
- `conversation_history` provides short-term memory (last 5 interactions) within a session
- Each tool's inputs and outputs are clearly defined and isolated

