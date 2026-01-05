# Parquet Agent v1.0 — Procedural Build Plan

> **Source of Truth**: `instructions/` folder  
> **No deviations. No additions. Execute as specified.**

---

## Grounding References

| Document | Contains |
|----------|----------|
| `instructions_plan.md` | Architecture, prompts, schemas (B1-B6), tables (1-10), epics |
| `mcp_tools_implementation.md` | Tool implementations, schemas, domain config |
| `controller_loop_pseudocode.md` | Orchestration skeleton |

---

# PHASE 1: Contracts & Schemas

**Source**: `instructions_plan.md` lines 1814-1818 (Epic 1)

## 1.1 Create JSON Schema Files

Create these files in `external/schemas/`:

| File | Source |
|------|--------|
| `query_spec.schema.json` | instructions_plan.md B1 (lines 1953-2028) |
| `query_spec_status.schema.json` | instructions_plan.md B2 (lines 2033-2077) |
| `investigation_plan_step.schema.json` | instructions_plan.md B3 (lines 2081-2109) |
| `executor_report.schema.json` | instructions_plan.md B4 (lines 2113-2150) |
| `policy_limits.schema.json` | instructions_plan.md B5 (lines 2154-2170) |
| `decider_output.schema.json` | instructions_plan.md B6 (lines 2174-2237) |

## 1.2 Create Schema Validators

```python
# external/agent/schema_validators.py

import jsonschema

def validate_decider_output(output: dict) -> bool:
    """Validate against decider_output.schema.json"""
    
def validate_executor_report(report: dict) -> bool:
    """Validate against executor_report.schema.json"""
    
def validate_query_spec(spec: dict) -> bool:
    """Validate against query_spec.schema.json"""
```

## 1.3 Create TypedDicts

**Source**: `instructions_plan.md` lines 76-84, `controller_loop_pseudocode.md` lines 16-29

```python
# external/agent/state_types.py

class ExecutorState(TypedDict):
    query_spec: dict
    query_spec_status: dict
    investigation_plan: list
    final_sql: str | None
    results: Any | None
    executor_report: dict
    policy_limits: dict

class ControllerState(TypedDict):
    user_query: str
    conversation_history: list
    domain_md: str
    policy_limits: dict
    query_spec: dict
    query_spec_status: dict
    last_executor_report: Optional[dict]
    attempt_count: int
```

---

# PHASE 2: Tools Layer

**Source**: `mcp_tools_implementation.md` Section 3

## 2.1 Create Tool Directory

```
external/tools/parquet_agent/
├── __init__.py
├── list_dir.py
├── inspect_table.py
├── preview_rows.py
├── search_glossary.py
├── nl_to_sql_planner.py
├── sql_plan_updater.py
├── query_safety_validator.py
├── execute_sql.py
└── query_result_evaluator.py
```

## 2.2 Adapt Existing Tools (5 tools)

| Tool | Source Code | Action | Reference |
|------|-------------|--------|-----------|
| `list_dir` | `duckdb_list_files` | Wrap with spec schema | mcp_tools_implementation.md 3.1 |
| `inspect_table` | `duckdb_describe_table` | Wrap with spec schema | mcp_tools_implementation.md 3.2 |
| `execute_sql` | `duckdb_query` | Wrap with spec schema | mcp_tools_implementation.md 3.8 |
| `query_safety_validator` | `external/tools/query_safety_validator.py` | Align I/O schema | mcp_tools_implementation.md 3.7 |
| `query_result_evaluator` | `external/tools/query_result_evaluator.py` | Align I/O schema | mcp_tools_implementation.md 3.9 |

### Schema Changes Required:

**`query_safety_validator`**:
- Input: `query` → `sql`, add `policy_limits`
- Output: `is_safe` → `allowed`, `reason` → `flags`

**`query_result_evaluator`**:
- Input: add `query_spec`, `results_summary`, `validation_checks`
- Output: `satisfied`, `issues`, `notes`

## 2.3 Build New Tools (3 tools)

| Tool | Implementation | Reference |
|------|----------------|-----------|
| `preview_rows` | DuckDB LIMIT query | mcp_tools_implementation.md 3.3 |
| `search_glossary` | Parse domain.md metrics/entities | mcp_tools_implementation.md 3.4 |
| `sql_plan_updater` | LLM-based SQL patching | mcp_tools_implementation.md 3.6 |

## 2.4 Rewrite Tool (1 tool)

| Tool | Reason | Reference |
|------|--------|-----------|
| `nl_to_sql_planner` | Wrong signature | mcp_tools_implementation.md 3.5 |

**Current signature** (wrong):
- Input: `query`, `table_schema`
- Output: `plan`, `plan_quality`, `clarification_questions`

**Required signature** (spec):
- Input: `query_spec`, `query_spec_status`
- Output: `sql` (string only)

## 2.5 Create Tool Config Files

Create in `config/tools/`:

| File | Reference |
|------|-----------|
| `list_dir.json` | mcp_tools_implementation.md Section 3.1 |
| `inspect_table.json` | mcp_tools_implementation.md Section 3.2 |
| `preview_rows.json` | mcp_tools_implementation.md Section 3.3 |
| `search_glossary.json` | mcp_tools_implementation.md Section 3.4 |
| `nl_to_sql_planner.json` | mcp_tools_implementation.md Section 3.5 |
| `sql_plan_updater.json` | mcp_tools_implementation.md Section 3.6 |
| `query_safety_validator.json` | mcp_tools_implementation.md Section 3.7 |
| `execute_sql.json` | mcp_tools_implementation.md Section 3.8 |
| `query_result_evaluator.json` | mcp_tools_implementation.md Section 3.9 |

## 2.6 Populate Domain Config

**File**: `domain_instructions/ecomm_domain.md`

**Source**: Already created per mcp_tools_implementation.md Section 4

---

# PHASE 3: Executor Graph

**Source**: `instructions_plan.md` lines 1832-1836 (Epic 4)

## 3.1 Graph Shape (6 nodes, linear)

**Source**: `instructions_plan.md` lines 26-42

```
START → InvestigationNode → SQLGenerationNode → SafetyValidationNode → ExecutionNode → EvaluationNode → OutcomeNode → END
```

## 3.2 Node Implementations

**Source**: `instructions_plan.md` lines 47-69

### InvestigationNode (Python)
- Inputs: `investigation_plan`, `query_spec`, `query_spec_status`
- Runs tools exactly as listed in plan
- Patches `query_spec` + `status`
- Fails fast if required gap can't be closed

### SQLGenerationNode (LLM)
- Uses `nl_to_sql_planner` tool
- **Gate**: Must check `spec_ready_for_sql()` first
- Output: `final_sql`

### SafetyValidationNode (Python)
- Uses `query_safety_validator` tool
- Enforces `policy_limits`
- Blocks or proceeds

### ExecutionNode (Python)
- Uses `execute_sql` tool
- Captures raw results or errors

### EvaluationNode (Python)
- Uses `query_result_evaluator` tool
- Checks results against `validation_checks`

### OutcomeNode (Python)
- Builds `ExecutorReport` (SUCCESS or ERROR)
- Uses schema from B4

## 3.3 SQL Generation Gate

**Source**: `instructions_plan.md` lines 1820-1824 (Epic 2)

```python
def spec_ready_for_sql(query_spec: dict, query_spec_status: dict, domain_md: str) -> bool:
    """
    Required fields must be verified/defaulted (not missing/conflict):
    - business_question
    - start_table_grain
    - time (unless rule == "no_time")
    - metrics (unless listing_allows_empty_metrics == true)
    """
```

**Domain check** (from `instructions_plan.md` lines 2906-2924):
```python
def metrics_empty_allowed(query_spec: dict, domain_md: str) -> bool:
    allow = get_domain_bool(domain_md, "listing_allows_empty_metrics")
    if allow is not True:
        return False
    return (
        isinstance(query_spec.get("metrics"), list)
        and len(query_spec["metrics"]) == 0
        and "listing" in (query_spec.get("aggregation_plan", "").lower())
    )
```

## 3.4 Early Halt Mechanism

**Source**: `instructions_plan.md` lines 1834-1836

- If any node fails → set `halt_execution = True`
- Downstream nodes become no-op
- OutcomeNode produces ERROR report

## 3.5 Wire LangGraph

```python
def run_executor(decider_output: dict, state: ControllerState) -> dict:
    """
    Runs Executor subgraph exactly once.
    Returns schema-valid executor_report.
    """
```

---

# PHASE 4: Decider

**Source**: `instructions_plan.md` lines 1838-1842 (Epic 5)

## 4.1 Register Decider Prompt

**Source**: `instructions_plan.md` lines 623-865

Full prompt text is in the instructions. Register in prompt registry.

## 4.2 Implement run_decider()

**Source**: `instructions_plan.md` lines 506-516

```python
def run_decider(state: ControllerState) -> dict:
    """
    Single LLM call using Decider prompt.
    Must validate output against decider_output.schema.json.
    If schema invalid, re-prompt internally (max 2 retries).
    """
```

## 4.3 Action Router

**Source**: `instructions_plan.md` lines 533-546

```python
if action == "ASK_USER": return render_ask_user(decider_output)
if action == "EXECUTE": return run_executor(decider_output, state)
if action == "BLOCK": return render_block(decider_output)
```

---

# PHASE 5: Controller Loop

**Source**: `controller_loop_pseudocode.md` (entire file)

## 5.1 Implement handle_query()

```python
def handle_query(user_query: str, conversation_history: list, prior_state: Optional[ControllerState] = None) -> dict:
    """
    Controller orchestrates:
      Decider -> Executor -> (ERROR -> Decider retry) until SUCCESS / ASK_USER / BLOCK / max_attempts.
    """
```

## 5.2 State Management

- `attempt_count`: lives in ControllerState, controller-owned
- `last_executor_report`: stored after each executor run
- `max_attempts`: enforced before each executor run

## 5.3 Render Functions

**Source**: `controller_loop_pseudocode.md` lines 53-82

- `render_ask_user(decider_output) -> dict`
- `render_block(decider_output) -> dict`
- `render_success(executor_report) -> dict`
- `render_error_max_attempts(last_report, attempt_count, max_attempts) -> dict`

---

# PHASE 6: Testing

**Source**: `instructions_plan.md` lines 1849-1860 (Epic 7-8)

## 6.1 Tool Unit Tests

**Source**: `mcp_tools_implementation.md` Section 6

| Tool | Test Case |
|------|-----------|
| `list_dir` | `{"path": "ECommerce"}` → 3 files |
| `inspect_table` | `{"path": "ECommerce/sample_sales_data.csv"}` → columns include order_id |
| `preview_rows` | `{"path": "...", "limit": 5}` → 5 rows |
| `search_glossary` | `{"term": "revenue", "domain": "ecomm"}` → hit |
| `nl_to_sql_planner` | Valid QuerySpec → SQL string |
| `sql_plan_updater` | SQL + patch → patched SQL |
| `query_safety_validator` | SELECT → allowed; DELETE → blocked |
| `execute_sql` | Valid SQL → results |
| `query_result_evaluator` | Results + spec → satisfied/issues |

## 6.2 Schema Validation Tests

- `validate_decider_output()` with valid/invalid JSON
- `validate_executor_report()` with SUCCESS/ERROR variants
- `validate_query_spec()` with complete/incomplete specs

## 6.3 Integration Tests

- Full flow: user query → ASK_USER response
- Full flow: user query → SUCCESS with SQL + results
- Full flow: user query → ERROR → retry → SUCCESS
- Full flow: user query → max_attempts reached

## 6.4 Failure Mode Tests

**Source**: `instructions_plan.md` line 1860

- Missing required fields → proper error
- Schema mismatch → validation failure
- Empty results → EMPTY error type
- Policy violation → POLICY error type
- Timeout → proper handling

---

# Execution Order

```
Week 1:
├── PHASE 1: Schemas (1 day)
├── PHASE 2.2: Adapt 5 tools (2 days)
├── PHASE 2.3: Build 3 new tools (2 days)
└── PHASE 2.4: Rewrite nl_to_sql_planner (1 day)

Week 2:
├── PHASE 3: Executor graph (3 days)
├── PHASE 4: Decider (2 days)
└── PHASE 5: Controller loop (1 day)

Week 3:
└── PHASE 6: Testing (5 days)
```

---

# Constraints (from instructions)

| Rule | Source |
|------|--------|
| Decider has 0 tools | instructions_plan.md line 201 |
| Executor has 9 tools | instructions_plan.md lines 236-300 |
| No fallbacks | instructions_plan.md line 99-103 |
| No self-repair | instructions_plan.md line 164 |
| No branching in Executor | instructions_plan.md line 92 |
| Schema validation required | instructions_plan.md line 397 |
| max_attempts enforced by controller | controller_loop_pseudocode.md line 119 |

---

# Files to Create/Modify

## Create New

| Path | Purpose |
|------|---------|
| `external/schemas/*.json` | 6 JSON schema files |
| `external/agent/schema_validators.py` | Validation functions |
| `external/agent/state_types.py` | TypedDicts |
| `external/tools/parquet_agent/*.py` | 9 tool implementations |
| `config/tools/*.json` | 9 tool config files |

## Modify Existing

| Path | Action |
|------|--------|
| `external/agent/parquet_agent.py` | Refactor to new architecture |
| `external/agent/planner_state.py` | Align with new state model |

---

# Validation Checkpoints

Before proceeding to next phase, verify:

| Phase | Checkpoint |
|-------|------------|
| 1 | All 6 schemas validate sample JSON |
| 2 | All 9 tools callable and return spec-compliant output |
| 3 | Executor graph runs end-to-end with mock data |
| 4 | Decider produces schema-valid output |
| 5 | Controller handles all 3 actions correctly |
| 6 | All tests pass |

