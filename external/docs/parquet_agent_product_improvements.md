# Parquet Agent — Product Improvement Recommendations (Implementation Paths)

Last updated: 2025-12-28

This document summarizes recommended product improvements for the Parquet Agent + its infra (DB/SSE/UI), and the most direct implementation path **aligned with the current codebase**.

---

## 4) Dataset discovery contract (agent folder + subfolders)

### Current behavior
- Agent scoping exists:
  - `agent.data_folder` is loaded into controller state (`agent_data_folder`)
  - executor normalizes tool paths via `normalize_data_path` in `external/agent/executor_nodes.py`
- But SQL view registration is shallow:
  - `external/tools/parquet_agent/execute_sql.py` only registers views for files in `base_path/<domain_dir>/*.csv|*.parquet`.
  - Deep subfolders won’t be registered as views → “table not found” style failures.

### Recommended paths
- **Option A (recommended): recursive discovery**
  - Update `execute_sql.py` to recursively discover `*.csv`/`*.parquet` under the scoped folder.
  - Keep view naming deterministic to avoid collisions.
- **Option B: manifest-based datasets**
  - Extend agent config to list datasets with explicit `{view_name, path}`.
  - `execute_sql.py` registers only those (predictable, faster, enterprise-friendly).
- **Option C: enforce flat folder**
  - Validate folder layout at agent creation time; block nested layouts.

---

## 5) Thinking + streaming (what’s feasible with current plumbing)

### Current behavior
- Decider supports “thinking trace” (see `external/agent/decider.py`) but the SSE flow doesn’t emit it.
- UI shows coarse progress only: `decider_done`, `sql_generated`, `results_ready`, `final_response`.

### Recommended paths
- **Low-effort, high-value (recommended first): stream artifacts (not tokens)**
  - Emit SSE events for:
    - `decider_thinking` (payload: thinking trace)
    - `decider_output` (payload: key spec fields)
    - `sql_generated` (already)
    - `results_ready` with a preview table payload (optional)
  - Update `web/templates/agent_chat.html` to render these progressively (collapsible sections).
- **True token streaming**
  - Requires streaming support from LLM client and chunked SSE events.
  - Start with streaming only the final commentary (prose) rather than Decider (strict JSON).

---

## 6) Intent router (“sky is blue” should not behave like a data query)

### Current behavior
- The parquet agent is a structured data-query pipeline; general questions aren’t a fit.

### Recommended paths
- Add a lightweight router step in the run start path (before Decider/Executor):
  - heuristic classifier or small LLM router
- Route to:
  - parquet data agent, OR
  - general chat agent, OR
  - ASK_USER prompting for dataset/metric/timeframe

---

## 7) DB-backed product UX (history + agent dropdown)

### What DB enables now
- You already persist:
  - `conversations`, `messages`, `runs`, `run_events`, `run_results`

### Recommended path
- **Conversation history UI**
  - Backend endpoints:
    - list conversations by user/agent
    - get messages for a conversation
  - Frontend:
    - sidebar to select conversation → set `conversation_id` and load messages
- **Agent dropdown**
  - Backend endpoint to list agents from DB (admin/user)
  - Frontend dropdown to switch agent context (and optionally conversation context)

---

## Suggested implementation order (fastest impact first)
1. Persist & rehydrate structured state (Decider output, richer history payload)
2. Make evaluator non-fatal (ASK_USER / success-with-warnings)
3. Max-attempt recovery card + quick replies
4. Dataset discovery contract (recursive or manifest)
5. Stream artifacts (thinking/spec/sql/table) over SSE
6. DB-backed history + agent switching UX
7. Intent router for non-data questions


