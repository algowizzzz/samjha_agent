- **Executor — LangGraph setup (concise, production-ready)**
    
    ### **Executor — what it does (only)**
    
    - **Consumes the execution packet** from the Decider (query spec, status, investigation plan, limits).
    - **Runs investigations** exactly as specified (schema lookup, glossary resolution, data checks).
    - **Generates SQL** *only after* required spec fields are complete.
    - **Validates safety** (policy, limits, cost, forbidden patterns).
    - **Executes SQL deterministically** (no guessing, no replanning).
    - **Evaluates results** (sanity checks, satisfaction against spec).
    - **Reports outcome** back to Decider:
        - `SUCCESS` with final output, or
        - `ERROR` with precise failure type + minimal fix suggestion.
    
    **One line:**
    
    > The Executor is a deterministic worker: it follows the contract, runs tools, checks results, and reports—nothing more.
    > 
    
    ### **Executor — LangGraph setup (concise, production-ready)**
    
    ---
    
    ## Graph shape (Executor only)
    
    ```
    START
      ↓
    InvestigationNode
      ↓
    SQLGenerationNode
      ↓
    SafetyValidationNode
      ↓
    ExecutionNode
      ↓
    EvaluationNode
      ↓
    OutcomeNode
      ↓
    END  (or RETRY edge back to Decider, outside this graph)
    
    ```
    
    ---
    
    ## Nodes (what each does)
    
    1. **InvestigationNode (Python)**
        - Inputs: `investigation_plan`, `query_spec`, `query_spec_status`
        - Runs tools *exactly as listed*
        - Patches `query_spec` + `status`
        - Fails fast if a required gap can’t be closed
    2. **SQLGenerationNode (LLM)**
        - Uses `nl_to_sql_planner`
        - Preconditions: required spec fields verified
        - Output: `final_sql`
    3. **SafetyValidationNode (Python)**
        - Enforces limits, forbidden patterns, cost guards
        - Blocks or proceeds
    4. **ExecutionNode (Python)**
        - Runs `execute_sql`
        - Captures raw results/errors
    5. **EvaluationNode (Python)**
        - Runs `query_result_evaluator`
        - Checks sanity vs `validation_checks`
    6. **OutcomeNode (Python)**
        - Builds **Executor Outcome Report** (SUCCESS / ERROR)
        - Includes minimal fix suggestion on error
    
    ---
    
    ## Shared state (Executor graph)
    
    ```python
    class ExecutorState(TypedDict):
        query_spec: dict
        query_spec_status: dict
        investigation_plan: list
        final_sql: str | None
        results: Any | None
        executor_report: dict
        policy_limits: dict
    
    ```
    
    ---
    
    ## Edges (LangGraph)
    
    - Linear edges between nodes
    - **No branching inside Executor**
    - Retry decision handled **outside** (controller calls Decider again)
    
    ---
    
    ## Key rules (Executor-only)
    
    - ❌ No ASK_USER
    - ❌ No replanning
    - ❌ No tool discovery
    - ✅ Only tools explicitly allowed
    - ✅ Deterministic, auditable steps
    
    ---
    
    ## One-line mental model
    
    > Executor = linear pipeline that closes gaps, runs SQL safely, and reports truth or failure.
    > 
    
    ---
    
    ## How prompts work in the Executor
    
    ### **There are TWO prompts, not one.**
    
    ---
    
    ## 1️⃣ SQL Generation Prompt (LLM) — **one prompt**
    
    **Exists as:**
    
    `nl_to_sql_planner` prompt
    
    **Where it lives:**
    
    - Prompt registry (same as today)
    - Used **only** in `SQLGenerationNode`
    
    **What it does:**
    
    - Converts a **completed Query Spec** into SQL
    - No planning, no guessing, no repair
    
    **When it runs:**
    
    - After InvestigationNode verifies required fields
    
    ---
    
    ## 2️⃣ Result Evaluation Prompt (LLM or rules) — **optional**
    
    **Exists as:**
    
    `query_result_evaluator` prompt (or pure Python)
    
    **Where it lives:**
    
    - Prompt registry (optional)
    - Used **only** in `EvaluationNode`
    
    **What it does:**
    
    - Checks whether results satisfy the Query Spec
    - Performs sanity checks
    
    ---
    
    ## What does **not** exist in Executor
    
    - ❌ No Decider prompt
    - ❌ No multi-step agent prompt
    - ❌ No “self-repair” prompt
    - ❌ No ASK_USER prompt
    
    ---
    
    ## Mental model
    
    > Decider = 1 prompt (writes the contract)
    > 
    > 
    > **Executor = 1–2 prompts (implements + evaluates the contract)**
    > 
    
    That’s it.
    
    Short answer: **it’s not explicitly enumerated in the canvas yet** — and that’s the last missing clarity.
    
    Below is the **clean, enforceable split** that matches everything you’ve designed.
    
    ---
    
    ## Tool ownership model (final, simple, non-confusing)
    
    ### Principle (very important)
    
    > Decider reasons. Executor touches the world.
    > 
    
    If a tool:
    
    - **changes state**, **reads data**, **executes SQL**, or **inspects files** → Executor
    - **analyzes, classifies, plans, or asks questions** → Decider (LLM-only, no tools)
    
    ---
    
    ## 1️⃣ Decider — tools it can use
    
    **Decider should use *NO data-access tools*.**
    
    It only uses **reasoning + static inputs**.
    
    ### Allowed inputs (not tools)
    
    - `domain.md`
    - User query
    - Conversation history
    - Previous `query_spec` + `query_spec_status`
    - Retry context from Executor
    
    ### Allowed actions
    
    - `ASK_USER(question)`
    - `EXECUTE(execution_packet)`
        
        (query_spec + investigation_plan)
        
    
    ### ❌ Tools Decider must NOT call
    
    - `list_dir`
    - `inspect_table`
    - `search_glossary`
    - `execute_sql`
    - `nl_to_sql`
    - validators
    
    **Why:**
    
    The moment Decider runs tools, you lose the clean boundary and re-introduce Gate/Executor confusion.
    
    ---
    
    ## 2️⃣ Executor — tools it can use (this is the important list)
    
    Executor owns **all operational tools**, grouped by purpose.
    
    ---
    
    ### A. Discovery & inspection tools
    
    *(Used only to fill missing checklist fields)*
    
    - `list_dir`
        - Purpose: table discovery
        - Fills: `query_spec.start_table` (candidates)
    - `inspect_table`
        - Purpose: schema, grain inference
        - Fills:
            - columns inventory
            - identifiers
            - grain (verified vs inferred)
    - `preview_rows` *(or `execute_sql LIMIT 10`)*
        - Purpose: row meaning, nulls, flags
        - Fills:
            - grain confidence
            - filters/status rules
    
    ---
    
    ### B. Glossary / semantics tools
    
    - `search_glossary`
        - Purpose: resolve ambiguous business terms
        - Fills:
            - metrics definitions
            - term → column mappings
        - Used **only if domain.md + schema are insufficient**
    
    ---
    
    ### C. SQL construction tools
    
    - `nl_to_sql_planner`
        - Input: **completed Query Spec**
        - Output: SQL draft
        - Executor must **not** call this if required checklist items are missing
    - `sql_plan_updater`
        - Purpose: bounded, mechanical fixes
        - Examples:
            - column rename
            - add LIMIT
            - apply missing default time filter
    
    ---
    
    ### D. Validation & execution tools
    
    - `query_safety_validator`
        - Purpose: policy + safety checks
        - Sets: `policy_flags`
    - `execute_sql`
        - Purpose: get results
        - Sets: `result_summary`
    - `query_result_evaluator`
        - Purpose: self-evaluation vs query + Query Spec
        - Sets: `result_evaluation`
    
    ---
    
    ## 3️⃣ Executor self-evaluation rule (key to your loop)
    
    Executor decides **only one thing**:
    
    > “Is the Query Spec satisfied by these results?”
    > 
    
    ### If YES
    
    → `FINISH`
    
    ### If NO
    
    → `RETURN_TO_DECIDER` with:
    
    - which checklist items failed
    - why they failed
    - suggested fix (if obvious)
    
    Executor **never asks the user**.
    
    ---
    
    ## 4️⃣ How this maps cleanly back to the canvas
    
    ### What’s already correct in the canvas
    
    - Decider produces:
        - Query Spec
        - Investigation Plan
    - Executor runs tools
    - Executor decides finish vs retry
    
    ### What’s missing (and now clarified)
    
    - **Decider has zero tools**
    - **Executor has all tools**
    - Tools exist only to **fill checklist gaps or verify risky assumptions**
    
    You can add this one small section to the canvas for completeness:
    
    ---
    
    ## Tool Ownership Summary (add this section)
    
    **Decider**
    
    - ❌ No tools
    - ✅ domain.md, reasoning, planning, ASK_USER
    
    **Executor**
    
    - ✅ list_dir
    - ✅ inspect_table
    - ✅ search_glossary
    - ✅ nl_to_sql_planner
    - ✅ sql_plan_updater
    - ✅ query_safety_validator
    - ✅ execute_sql
    - ✅ query_result_evaluator
    
    ---
    
    ## Final mental model (the one to keep)
    
    > Decider writes the checklist.Executor fills the checklist.SQL is just a by-product of a completed checklist.
    > 
    
    If you want next, I can:
    
    - add a **hard validation rule** (“executor cannot call nl_to_sql unless checklist complete”)
    - or convert this into a **unit-testable policy matrix** (tool × role enforcement)
- Decider
    
    ## 1) Treat the Decider as a compiler front-end
    
    **Input:** user_query + history + domain_md + prior spec/status + last executor report + policy limits
    
    **Output:** exactly one of:
    
    - `ASK_USER` (with a single best question + why it’s non-defaultable)
    - `EXECUTE` (with a complete execution packet)
    - `BLOCK` (cannot proceed even with user input, e.g., no data access)
    
    This matches your Table 3A/3B.
    
    ---
    
    ## 2) Make the Decider prompt “schema-first”
    
    Give the Decider one job: **fill/patch Query Spec + Query Spec Status + Investigation Plan**, then decide action.
    
    ### Enforce a JSON schema (hard)
    
    Implement a JSON schema validator in code. If it fails, you re-prompt the Decider with the validation errors (no user involvement).
    
    **Top-level Decider output shape:**
    
    ```json
    {
      "action": "ASK_USER|EXECUTE|BLOCK",
      "domain": "",
      "intent": "",
      "decisions": { "comprehension": "", "determinacy": "", "clarification_need": "" },
      "query_spec": { ...Table9... },
      "query_spec_status": { ...Table10... },
      "investigation_plan": [ ...Table4B... ],
      "expected_output": "",
      "stop_conditions": [],
      "ask_user": { "question": "", "why_non_defaultable": "", "what_answer_unblocks": "" },
      "block_reason": ""
    }
    
    ```
    
    ---
    
    ## 3) Use “gap-driven planning” (key engineering trick)
    
    The Decider should **never plan tools generically**. It should plan **only to close missing/unsafe spec fields**.
    
    Mechanism:
    
    1. Compute missing or unsafe fields using **Query Spec Status** (Table 10).
    2. Map each gap to one tool via **Tool Capability Cards** (Table 4A).
    3. Produce Investigation Plan (Table 4B) with 1–4 steps max.
    
    Example rule:
    
    - If `metrics` is missing → `search_glossary`
    - If `start_table.path` missing → `list_dir`
    - If `grain` missing → `inspect_table`
    - If required_minimum satisfied → `nl_to_sql_planner` is allowed (but that’s Executor-side)
    
    ---
    
    ## 4) Put strict boundaries into the prompt
    
    In the Decider prompt, add hard constraints:
    
    - **No SQL generation**
    - **No tool calls**
    - **Must emit** only the JSON packet or ASK_USER
    - **Must mark** each Query Spec field with status + source
    - **Must not default** “grain” or “time axis” unless domain_md explicitly provides a rule
    
    This prevents “plausible but wrong” execution.
    
    ---
    
    ## 5) Implement the runtime loop (simple, deterministic)
    
    In your orchestrator:
    
    1. Call Decider → get `action`.
    2. If `ASK_USER` → show question to user.
    3. If `BLOCK` → stop.
    4. If `EXECUTE` → send `execution_packet` to Executor.
    
    On Executor error:
    
    - Executor returns Outcome Report (Table 8B)
    - Orchestrator stores it as `last_executor_report`
    - Re-call Decider with the same user_query + `last_executor_report` (Decider chooses RETRY_WITH_PATCH / ASK / BLOCK)
    
    Also enforce `policy_limits.max_attempts`.
    
    ---
    
    ## 6) What you physically store in scratchpad
    
    Store only these as canonical state:
    
    - `prior_query_spec`
    - `prior_query_spec_status`
    - `last_executor_report` (optional)
    - `domain_md_ref` (which domain doc chosen)
    - `attempt_count`
    
    Everything else is derivable.
    
    ---
    
    ## 7) One-prompt Decider structure (recommended sections)
    
    Inside the single Decider prompt:
    
    1. **Role + boundaries**
    2. **Inputs (Table 1)**
    3. **Decision rubric (Table 2 updated)**
    4. **Output schemas (Table 3A/3B)**
    5. **Query Spec contract (Table 9)**
    6. **Spec status tracker rules (Table 10)**
    7. **Tool capability cards (Table 4A)**
    8. **Planning rule: “gap → tool → success condition”**
    9. **Examples: one ASK_USER, one EXECUTE**
    
    ---
    
    If you want, I can write the **full single Decider prompt** in production form (with strict JSON schema, validation rules, and a couple of canonical examples) so you can drop it into your prompt registry.
    
    ---
    
    ### 1️⃣ Decider Orchestrator (core)
    
    **What:**
    
    A function that calls the Decider prompt and validates its output.
    
    **You write:**
    
    - `run_decider(input_context) -> decider_output`
    - JSON Schema validator (hard fail + re-prompt on error)
    
    ---
    
    ### 2️⃣ Query Spec & Status Models
    
    **What:**
    
    Typed schemas for Tables 9 & 10.
    
    **You write:**
    
    - `QuerySpec` dataclass / Pydantic model
    - `QuerySpecStatus` dataclass / Pydantic model
    - `validate_required_minimum(spec, status)`
    
    ---
    
    ### 3️⃣ Action Router
    
    **What:**
    
    Routes Decider output.
    
    **You write:**
    
    ```python
    if action == "ASK_USER": return ask_user()
    if action == "EXECUTE": return run_executor(packet)
    if action == "BLOCK": return stop()
    
    ```
    
    ---
    
    ### 4️⃣ Investigation Plan Executor (Executor-side)
    
    **What:**
    
    Runs tools in the order Decider specifies.
    
    **You write:**
    
    - Loop over `investigation_plan`
    - Tool dispatch map
    - Capture patches → update `query_spec` + `status`
    
    ---
    
    ### 5️⃣ Executor Wrapper
    
    **What:**
    
    Controlled SQL execution + validation.
    
    **You write:**
    
    - Tool registry (allowed tools only)
    - SQL safety validator
    - Result evaluator
    - Outcome report generator (Table 8)
    
    ---
    
    ### 6️⃣ Retry Loop Controller
    
    **What:**
    
    Handles REPAIR / retry logic.
    
    **You write:**
    
    - Store `last_executor_report`
    - Increment attempt count
    - Re-invoke Decider with patch context
    - Enforce `policy_limits.max_attempts`
    
    ---
    
    ### 7️⃣ Persistent Scratchpad
    
    **What:**
    
    Canonical state between turns.
    
    **You write:**
    
    - Save/load:
        - `prior_query_spec`
        - `prior_query_spec_status`
        - `last_executor_report`
        - `attempt_count`
    
    ---
    
    ### In one line
    
    > You need schema validation, routing, tool dispatch, and state persistence — not more prompts.
    > 
    
    If you want, next I can give a **~200-line reference implementation skeleton**.
    
    Below is the **full, production-ready single Decider prompt**.
    
    It is **schema-driven, strict, and executable** as-is in a prompt registry.
    
    ---
    
    # **DECIDER (Gate) — Single Canonical Prompt**
    
    ## ROLE
    
    You are the **Decider (Gate)** in a two-stage system.
    
    Your job is to **decide whether and how a query can be executed**, not to execute it.
    
    You **do not**:
    
    - Generate SQL
    - Call tools
    - Inspect data directly
    
    You **do**:
    
    - Produce a **truth contract** (Query Spec)
    - Track **what is known vs assumed vs missing**
    - Either **ASK_USER**, **EXECUTE**, or **BLOCK**
    
    ---
    
    ## HARD CONSTRAINTS (NON-NEGOTIABLE)
    
    1. **Output must be valid JSON only** (no prose).
    2. You must output **exactly one action**: `ASK_USER`, `EXECUTE`, or `BLOCK`.
    3. You must fill **Query Spec (Table 9)** and **Query Spec Status (Table 10)**.
    4. You must **not invent facts** about data, metrics, or schemas.
    5. You must **not default**:
        - `grain`
        - `time.column`
        - `metrics`
            
            unless an explicit rule exists in `domain_md`.
            
    6. If a required item is missing and non-defaultable → `ASK_USER` or `BLOCK`.
    
    ---
    
    ## INPUTS (READ-ONLY)
    
    You are given:
    
    - `user_query`
    - `conversation_history`
    - `domain_md`
    - `prior_query_spec`
    - `prior_query_spec_status`
    - `last_executor_report` (optional)
    - `policy_limits`
    
    ---
    
    ## DECISION RUBRIC (YOU MUST FOLLOW)
    
    ### Step 1 — Comprehension
    
    - If the question is **unintelligible** → `ASK_USER`
    - If intelligible, continue
    
    ### Step 2 — Determinacy
    
    - If multiple interpretations **change the answer materially** and no safe default exists → `ASK_USER`
    - Else continue
    
    ### Step 3 — Fill / Patch Query Spec
    
    - Populate Query Spec (Table 9)
    - For each item, set status in Query Spec Status (Table 10):
        - `missing`
        - `defaulted`
        - `inferred`
        - `verified`
    - Record **source** for each item
    
    ### Step 4 — Evidence Sufficiency
    
    - If available datasets **cannot support grain + metrics** → `BLOCK`
    
    ### Step 5 — Decide Action
    
    - If required minimum is satisfied → `EXECUTE`
    - Else if user can resolve → `ASK_USER`
    - Else → `BLOCK`
    
    ---
    
    ## REQUIRED MINIMUM FOR EXECUTION
    
    All must be **verified or user-approved defaulted**:
    
    - `business_question`
    - `start_table.path`
    - `grain`
    - `time.column` *(or explicit “no_time” rule)*
    - `metrics`
    
    ---
    
    ## TOOL CAPABILITY CARDS (READ-ONLY KNOWLEDGE)
    
    You plan investigations **by capability**, not syntax.
    
    | Tool | Can Fill | Cannot Do |
    | --- | --- | --- |
    | list_dir | start_table.path | infer schema |
    | inspect_table | grain (candidate), time.column (candidate), columns | define metric meaning |
    | search_glossary | metric semantics | verify data exists |
    | nl_to_sql_planner | SQL generation (Executor only) | choose missing contracts |
    
    You **do not call tools**.
    
    You only produce an **Investigation Plan**.
    
    ---
    
    ## OUTPUT SCHEMA (STRICT)
    
    ### Top-Level
    
    ```json
    {
      "action": "ASK_USER | EXECUTE | BLOCK",
      "domain": "",
      "intent": "",
      "decisions": {
        "comprehension": "INTELLIGIBLE | UNINTELLIGIBLE",
        "determinacy": "DETERMINED | UNDERDETERMINED",
        "clarification_need": "ASK_REQUIRED | DEFAULT_OK"
      },
      "query_spec": { },
      "query_spec_status": { },
      "investigation_plan": [],
      "expected_output": "",
      "stop_conditions": [],
      "ask_user": {
        "question": "",
        "why_non_defaultable": "",
        "what_answer_unblocks": ""
      },
      "block_reason": ""
    }
    
    ```
    
    ---
    
    ## QUERY SPEC (TABLE 9)
    
    ```json
    {
      "business_question": "",
      "output_shape": {
        "type": "",
        "columns": []
      },
      "start_table": {
        "name": "",
        "path": ""
      },
      "grain": "",
      "time": {
        "column": "",
        "rule": "",
        "n_days": null
      },
      "metrics": [],
      "dimensions": [],
      "filters": [],
      "joins": [],
      "aggregation_plan": "",
      "validation_checks": [],
      "performance_guardrails": [],
      "defaults_used": [],
      "open_questions": []
    }
    
    ```
    
    ---
    
    ## QUERY SPEC STATUS (TABLE 10)
    
    ```json
    {
      "business_question": { "status": "", "source": "", "notes": "", "blocks_execution": false },
      "output_shape": { "status": "", "source": "", "notes": "", "blocks_execution": false },
      "start_table_grain": { "status": "", "source": "", "notes": "", "blocks_execution": true },
      "time": { "status": "", "source": "", "notes": "", "blocks_execution": true },
      "metrics": { "status": "", "source": "", "notes": "", "blocks_execution": true },
      "dimensions": { "status": "", "source": "", "notes": "", "blocks_execution": false },
      "filters": { "status": "", "source": "", "notes": "", "blocks_execution": false },
      "joins": { "status": "", "source": "", "notes": "", "blocks_execution": true },
      "aggregation_plan": { "status": "", "source": "", "notes": "", "blocks_execution": true },
      "validation_checks": { "status": "", "source": "", "notes": "", "blocks_execution": false },
      "performance_guardrails": { "status": "", "source": "", "notes": "", "blocks_execution": false }
    }
    
    ```
    
    ---
    
    ## INVESTIGATION PLAN (TABLE 4B RULE)
    
    Only include steps that **close missing or unsafe fields**.
    
    Each step must specify:
    
    - tool
    - args
    - gap it fills
    - success condition
    
    Max **4 steps**.
    
    ---
    
    ## EXAMPLES (MINIMAL)
    
    ### Example: ASK_USER
    
    - Grain missing
    - No domain rule exists
    
    → Ask **one** precise question that unblocks execution.
    
    ### Example: EXECUTE
    
    - Required minimum satisfied
    - Investigation plan completes remaining inferred items
    
    ---
    
    ## FINAL CHECK
    
    Before output:
    
    - Is every blocking item either resolved or escalated?
    - Did you avoid inventing data?
    - Did you choose exactly one action?
    
    **Then output JSON only.**
    
- tables
    
    ## All tables you need
    
    1. **Decider Inputs** (what Decider reads)
    2. **Decider Decisions & Actions** (what Decider can output)
    3. **Decider Plan Packet** (the execution packet sent to Executor)
    4. **Tool Capability Cards** (how Decider “knows” tools well enough to plan efficiently)
    5. **Executor Inputs** (what Executor receives)
    6. **Executor Allowed Tools** (tool list + purpose)
    7. **Executor Run Log** (what Executor records per step)
    8. **Executor Outcome Report** (what Executor sends back to Decider on error/success)
    9. **Query Spec** (single checklist contract)
    10. **Query Spec Status** (gap-filling tracker: missing/defaulted/inferred/verified)
    
    Below are the templates with **empty fields to fill**.
    
    ---
    
    # Table 1: Decider Inputs
    
    | Field | Type | Source | Notes / What it’s for | Value |
    | --- | --- | --- | --- | --- |
    | user_query | str | User | Current request |  |
    | conversation_history | list | System | Last N turns |  |
    | domain_md | text/md | KB | Domain defaults + rules + starter plan |  |
    | prior_query_spec | dict | Scratchpad | Existing checklist values (if follow-up) |  |
    | prior_query_spec_status | dict | Scratchpad | Which items are missing/verified |  |
    | last_executor_report | dict | Executor | Only if retry/error happened |  |
    | policy_limits | dict | Config | max attempts, safe defaults |  |
    
    ---
    
    ---
    
    # Table 3: Decider Action Output Template
    
    ### 3A) If Decider asks user
    
    | Field | Value |
    | --- | --- |
    | action | ASK_USER |
    | question |  |
    | why_non_defaultable |  |
    | what_answer_unblocks |  |
    
    ### 3B) If Decider sends plan to executor
    
    | Field | Value |
    | --- | --- |
    | action | EXECUTE |
    | domain |  |
    | intent |  |
    | query_spec | (Table 9) |
    | query_spec_status | (Table 10) |
    | investigation_plan | (Table 4B) |
    | expected_output |  |
    | stop_conditions |  |
    
    ---
    
    ## 4B) Investigation plan (generated per query)
    
    | Step # | Tool | Args | Why (which gap it fills) | Success condition |
    | --- | --- | --- | --- | --- |
    | 1 |  |  |  |  |
    | 2 |  |  |  |  |
    | 3 |  |  |  |  |
    | 4 |  |  |  |  |
    
    ---
    
    # Table 5: Executor Inputs
    
    | Field | Type | Source | Value |
    | --- | --- | --- | --- |
    | execution_packet | dict | Decider |  |
    | query_spec | dict | Packet |  |
    | query_spec_status | dict | Packet |  |
    | investigation_plan | list | Packet |  |
    | policy_limits | dict | Config |  |
    
    ---
    
    # Table 6: Executor Allowed Tools
    
    | Tool | Allowed | Used for | Notes |
    | --- | --- | --- | --- |
    | list_dir | ✅ | Discover tables/files |  |
    | inspect_table | ✅ | Get schema / infer grain / time columns |  |
    | preview_rows | ✅ | Sample rows for grain/filter intuition | Limited to ≤100 rows |
    | search_glossary | ✅ | Resolve ambiguous business terms | Only if needed |
    | nl_to_sql_planner | ✅ | Generate SQL from completed checklist | Blocked if required fields missing |
    | sql_plan_updater | ✅ | *Only if Decider includes it in plan* | (Strict mode: no self-repair) |
    | query_safety_validator | ✅ | Safety/policy validation |  |
    | execute_sql | ✅ | Run SQL |  |
    | query_result_evaluator | ✅ | Check satisfaction |  |
    
    ---
    
    ---
    
    # Table 8: Executor Outcome Report
    
    ### 8A) Success
    
    | Field | Value |
    | --- | --- |
    | status | SUCCESS |
    | final_sql |  |
    | result_summary |  |
    | evaluation |  |
    | finished_output |  |
    
    ### 8B) Error / Retry (Executor returns to Decider)
    
    | Field | Value |
    | --- | --- |
    | status | ERROR |
    | error_type | SCHEMA / SQL / EMPTY / GRAIN / POLICY / NO_DATASET |
    | failed_checklist_items |  |
    | what_changed |  |
    | minimal_fix_suggestion |  |
    | last_sql |  |
    | last_error |  |
    
    ---
    
    ---
    
    ---
    
    ### Table 2: Decider Decisions and Actions (Updated)
    
    | Decision Area | Options | Rule of thumb | Selected |
    | --- | --- | --- | --- |
    | Domain | ecommerce / mr / ccr / generic | Choose best-matching `domain_md` |  |
    | Intent | NEW_QUERY / FOLLOW_UP / MODIFY / DRILL_DOWN / EXTEND | Based on history + `prior_query_spec` |  |
    | Comprehension | INTELLIGIBLE / UNINTELLIGIBLE | If unclear language/refs → UNINTELLIGIBLE |  |
    | Determinacy | DETERMINED / UNDERDETERMINED | If multiple meanings change result → UNDERDETERMINED |  |
    | Clarification Need | ASK_REQUIRED / DEFAULT_OK | Ask only if underdetermined + no safe default |  |
    | Action | ASK_USER / EXECUTE / BLOCK | Ask if user can resolve; Block if cannot |  |
    | Safety Mode | NORMAL / STRICT | STRICT if retries high / sensitive query |  |
    | Retry Context | NONE / RETRY_WITH_PATCH | Use when `last_executor_report` exists |  |
    
    ---
    
    ### Table 4A: Tool Capability Cards (Updated)
    
    | Tool | Purpose | Fills which checklist items | Preconditions | Cost | Risk | Typical failures | Cannot do |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | list_dir | Discover datasets/files | `start_table.path`, evidence discovery | Root path known | Low | Low | Wrong directory scope | Infer schema/columns |
    | inspect_table | Read schema + sample stats | `output_shape.columns`, `grain` (candidate), `time.column` (candidate) | Table path exists | Med | Low | Large tables / slow | Decide metric meaning |
    | search_glossary | Resolve business terms | `metric_contracts`, `filters` semantics | Domain selected | Low | Med | No match / conflicting defs | Verify data availability |
    | nl_to_sql_planner | Generate SQL from spec | `final_sql` (derived), join/agg draft | Query spec “ready” | Med | Med | Hallucinated joins | Choose missing contracts |
    | query_safety_validator | Check policy & guardrails | `performance_guardrails`, safety flags | SQL exists | Low | Low | Over-blocking | Fix SQL logic |
    | execute_sql | Run SQL | results | Valid SQL + connection | Med/High | Med | Timeouts / perms | Decide meaning |
    | query_result_evaluator | Check satisfaction/sanity | `validation_checks` outcomes | Results exist | Low | Med | False negatives | Repair SQL |
    | sql_plan_updater | Apply Decider-approved patch | `query_spec` patch | Patch provided | Low | Low | Patch conflicts | Invent new plan |
    
    ---
    
    ### Table 7: Executor Run Log (Updated)
    
    | Step # | Tool | Args | Result summary | validation_outcome (pass/fail) | validation_notes | query_spec_patch | query_spec_status_patch | assumptions_used |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 |  |  |  |  |  |  |  |  |
    | 2 |  |  |  |  |  |  |  |  |
    
    ---
    
    ### Table 9: Query Spec (Updated)
    
    | Item | Value |
    | --- | --- |
    | business_question |  |
    | output_shape.type |  |
    | output_shape.columns |  |
    | start_table.name |  |
    | start_table.path |  |
    | grain |  |
    | time.column |  |
    | time.rule |  |
    | time.n_days |  |
    | metrics |  |
    | dimensions |  |
    | filters |  |
    | joins |  |
    | aggregation_plan |  |
    | validation_checks |  |
    | performance_guardrails |  |
    | defaults_used |  |
    | open_questions |  |
    | **required_minimum** | **[business_question, start_table.path, grain, time.column (or explicit “no_time”), metrics]** |
    
    ---
    
    ### Table 10: Query Spec Status (Updated)
    
    | Item | status (missing/defaulted/inferred/verified) | source (domain_md/tool_result/user/rule) | notes | blocks_execution? | escalation_target |
    | --- | --- | --- | --- | --- | --- |
    | business_question |  |  |  | ✅/❌ | ASK_USER/BLOCK |
    | output_shape |  |  |  | ❌ |  |
    | start_table + grain |  |  |  | ✅/❌ | ASK_USER/BLOCK |
    | time |  |  |  | ✅/❌ | ASK_USER/BLOCK |
    | metrics |  |  |  | ✅/❌ | ASK_USER/BLOCK |
    | dimensions |  |  |  | ❌ |  |
    | filters |  |  |  | ❌ |  |
    | joins |  |  |  | ✅/❌ | REPAIR/ASK_USER |
    | aggregation_plan |  |  |  | ✅/❌ | REPAIR |
    | validation_checks |  |  |  | ❌ |  |
    | performance_guardrails |  |  |  | ❌ |  |
    
    ---
    
- scratchpad fields
    
    # Complete State Template v4 – Minimal Scratchpad + Strict No-Retry Executor
    
    **Date:** 2025-12-22
    
    **Source:** `external/agent/planner_state.py` (refactor target)
    
    **Model:** Two-role (Decider / Executor) with single contract (`query_spec`) and strict executor governance
    
    ---
    
    ## Design Principles (Authoritative)
    
    1. **Single contract**
    - `scratchpad.query_spec` is the checklist and the source of truth.
    1. **Gap-filling, not evidence collection**
    - Tools exist to **fill missing checklist fields** or **verify risky assumptions** (grain/time/join keys).
    - Track *how a field was resolved* via status + provenance.
    1. **Hard role boundary**
    - **Decider:** LLM-only reasoning, planning, ASK_USER
    - **Executor:** owns all tools, runs the exact plan once
    1. **Strict executor governance (no authority to rerun)**
    - Executor executes the Decider plan once and either finishes or reports back.
    
    ---
    
    ## Table 1: Scratchpad Fields (LLM-Visible, Minimal & Canonical)
    
    Goal: keep scratchpad **small, stable, and contract-first**.
    
    | # | Field Name | Type | Owner | Purpose |
    | --- | --- | --- | --- | --- |
    | 1 | `query_spec` | Dict | Decider → Executor | **Single checklist contract** (values) |
    | 2 | `query_spec_status` | Dict | Decider / Executor | Per-item `status/source/notes` |
    | 3 | `clarity_status` | Optional[str] | Decider | `clear |
    | 4 | `decider_packet` | Optional[Dict] | Decider | Last plan sent: domain/intent/tool sequence |
    | 5 | `last_executor_report` | Optional[Dict] | Executor | Minimal outcome + errors + failed checklist items |
    
    ### Removed from scratchpad (moved to backend cache/trace)
    
    - `tool_results` → backend `trace`
    - `datasets_considered` → backend `datasets_cache`
    - `candidate_tables` → backend `candidates_cache`
    - `mapping_context` → backend `mapping_cache`
    - `glossary_facts` → backend `glossary_hits` (optionally include last 1–2 hits in `last_executor_report.snippets`)
    - `sql_attempts`, `final_sql` → backend `sql_history`
    - `result_summary`, `result_evaluation` → backend `result_cache`
    - `policy_flags` → backend `policy_cache`
    
    ### Deleted (derive instead)
    
    - `query_related_info` (delete)
    - `missing_fields` (derive from `query_spec_status`)
    - `proposed_defaults` (store as `query_spec_status.*.notes` + `source=rule/domain_md`)
    
    ---
    
    ## Table 2: `query_spec` (Checklist – Single Contract)
    
    Required fields must be resolved (not `missing`) before SQL generation.
    
    | Key | Type | Required | Notes |
    | --- | --- | --- | --- |
    | `business_question` | str | ✅ | 1 sentence |
    | `output_shape` | Dict | ✅ | `{type: single_value |
    | `start_table` | Dict | ✅ | `{name, path}` |
    | `grain` | str | ✅ | “one row per …” |
    | `time` | Dict | ⚠️ Defaultable | `{column, rule, n_days?}` |
    | `metrics` | List[Dict] | ⚠️ | definitions + formulas if derived |
    | `dimensions` | List[Dict] | ⚠️ | group-by fields |
    | `filters` | List[Dict] | ⚠️ | include rationale |
    | `joins` | List[Dict] | ⚠️ | canonical joins only |
    | `aggregation_plan` | str | ✅ | where aggregation happens |
    | `validation_checks` | List[str] | ✅ | 2–3 checks |
    | `performance_guardrails` | Dict | ❌ | e.g., `{limit:50, avoid_select_star:true}` |
    
    ---
    
    ## Table 3: `query_spec_status` (Gap-Filling Tracker)
    
    Same keys as `query_spec`, but each key holds:
    
    ```json
    {
      "status": "defaulted | inferred | verified | missing",
      "source": "domain_md | tool_result | user | rule",
      "notes": "short rationale"
    }
    
    ```
    
    Purpose:
    
    - Enforce completeness
    - Explain assumptions
    - Drive minimal replans (checklist-driven)
    
    ---
    
    ## Table 4: PlannerState (Runtime / Backend Only)
    
    PlannerState holds **runtime, caches, traces, loop-control** (not the contract).
    
    | Field | Purpose |
    | --- | --- |
    | `user_query` | current query |
    | `conversation_history` | context window |
    | `domain` | selected domain |
    | `intent` | query intent |
    | `scratchpad` | minimal LLM contract |
    | `schema_snapshot` | schema cache |
    | `glossary_hits` | glossary cache |
    | `datasets_cache` | discovered datasets |
    | `candidates_cache` | scored candidates |
    | `mapping_cache` | mapping synthesis |
    | `sql_history` | sql drafts/outcomes |
    | `result_cache` | execution summaries |
    | `policy_cache` | validator flags |
    | `trace` | tool trace |
    | `reasoning_log` | full reasoning |
    | `error_state` | error context |
    | `sql_attempt_count` | loop control |
    | `repair_count` | loop control |
    
    Cleanup note:
    
    - Remove duplicated `candidate_tables` / `execution_result_summary` fields if they mirror caches above.
    - Remove `next_action_hint` unless actively used.
    
    ---
    
    ## Table 5: Tool Ownership (Enforced)
    
    ### Decider
    
    - **No tools** (LLM-only reasoning)
    - Inputs: user query, history, `domain.md`, scratchpad, `last_executor_report`
    - Outputs: `ASK_USER(question)` or `EXECUTE(decider_packet)`
    
    ### Executor
    
    Executor owns **all operational tools** and executes only what Decider specifies:
    
    - `list_dir`
    - `inspect_table`
    - `preview_rows` (or safe `execute_sql LIMIT` equivalent)
    - `search_glossary`
    - `nl_to_sql_planner`
    - `query_safety_validator`
    - `execute_sql`
    - `query_result_evaluator`
    - `sql_plan_updater` *(only if included in the plan)*
    
    ---
    
    ## Table 6: Tool Capability Cards (What Decider Needs to Plan Efficiently)
    
    Decider plans well when it receives a compact “capability card” per tool.
    
    Each card: `fills`, `preconditions`, `cost`, `use_when`, `output`.
    
    - **list_dir**
        - fills: `query_spec.start_table` candidates
        - preconditions: domain selected
        - cost: cheap
        - use_when: start_table missing OR datasets_cache empty
        - output: files/subdirs
    - **inspect_table**
        - fills: `grain`, required columns, identifiers, time candidates
        - preconditions: candidate selected
        - cost: medium
        - use_when: grain/time/required columns are missing or only inferred
        - output: columns + types
    - **preview_rows**
        - fills: status flags, null behavior, grain intuition
        - preconditions: table known
        - cost: medium
        - use_when: filter rules unclear or grain uncertain
        - output: 5–10 rows
    - **search_glossary**
        - fills: metric definitions, term mappings, canonical joins
        - preconditions: ambiguous term OR mapping missing
        - cost: medium
        - use_when: business term unresolved after schema inspection
        - output: hits
    - **nl_to_sql_planner**
        - fills: SQL draft
        - preconditions: required `query_spec` fields resolved
        - cost: medium
        - use_when: checklist complete
        - output: SQL
    - **query_safety_validator**
        - fills: policy flags
        - preconditions: SQL exists
        - cost: cheap
        - use_when: always before execute
        - output: flags
    - **execute_sql**
        - fills: result summary
        - preconditions: SQL validated
        - cost: expensive
        - use_when: run final query
        - output: rows + metadata
    - **query_result_evaluator**
        - fills: satisfaction + issues
        - preconditions: results exist
        - cost: cheap
        - use_when: always after execute
        - output: issues + suggestions
    - **sql_plan_updater**
        - fills: minor SQL patch
        - preconditions: known error type in report
        - cost: cheap
        - use_when: only if Decider includes it
        - output: patched SQL
    
    ---
    
    ## Table 7: `decider_packet` (Decider → Executor Plan Contract)
    
    ```json
    {
      "domain": "ecommerce",
      "intent": "NEW_QUERY | FOLLOW_UP | MODIFY | DRILL_DOWN | EXTEND",
      "tool_sequence": [
        {"tool": "list_dir", "args": {"path": "ecommerce"}, "why": "discover tables"},
        {"tool": "inspect_table", "args": {"path": ".../products.csv"}, "why": "confirm columns + grain"}
      ],
      "stop_conditions": {
        "require_query_spec_complete": true,
        "max_tools": 6
      }
    }
    
    ```
    
    Planning rule for retries:
    
    - Plan must be **minimal**: target only the failed checklist items.
    
    ---
    
    ## Table 8: `last_executor_report` (Executor → Decider Return Contract)
    
    Executor executes the plan once and returns **either** SUCCESS or ERROR.
    
    ```json
    {
      "status": "SUCCESS | ERROR",
      "error_type": "SCHEMA | SQL | EMPTY | GRAIN | POLICY | NO_DATASET | AMBIGUOUS",
      "failed_checklist_items": ["start_table", "grain", "time", "joins"],
      "message": "short summary",
      "suggested_minimal_fix": "suggestion (not a plan)",
      "snippets": {
        "last_tool": "inspect_table",
        "last_error": "column not found: title",
        "candidate_table": "products",
        "schema_cols": ["product_id", "name", "category"],
        "sql": "SELECT ..."
      }
    }
    
    ```
    
    ---
    
    ## Table 9: Hard Constraints (Non-Negotiable)
    
    1. **Executor cannot rerun or extend**
    - Executes exactly `decider_packet.tool_sequence` once.
    1. **Executor cannot ask the user**
    - Any non-defaultable ambiguity returns to Decider.
    1. **Decider cannot call tools**
    - Plans using only `domain.md`, scratchpad, and last_executor_report.
    1. **SQL generation gate**
    - Executor must not call `nl_to_sql_planner` unless required `query_spec` keys are not `missing`.
    1. **Retry logic is checklist-driven**
    - Reports must include `failed_checklist_items`.
    
    ---
    
    ## Outcome
    
    This v4 template provides:
    
    - A minimal, stable scratchpad
    - A single contract (`query_spec`)
    - Efficient, minimal replanning
    - Strict executor governance (no retries without Decider)
    
    Yes — and the clean way to do this is to make a **single template that the Decider fills from `domain.md`** using three fields per item:
    
    - **Value** (what we set right now)
    - **Rationale** (which part of `domain.md` justified it)
    - **Comment** (confidence + what Executor must verify / fill)
    
    Below is a **drop-in template** + an **example filled for an ecommerce domain** (based on a typical `ecommerce_domain.md` playbook structure).
    
    ---
    
    ## Decider Bootstrap Fill Template (from `domain.md`)
    
    ### Query Spec (pre-filled by Decider)
    
    | Query Spec Item | Value (from domain.md) | Rationale (from domain.md) | Comment (confidence + what Executor verifies) |
    | --- | --- | --- | --- |
    | Business question |  |  |  |
    | Output shape + required columns |  |  |  |
    | Start table + grain |  |  |  |
    | Time column + default timeframe rule |  |  |  |
    | Metrics (definitions) |  |  |  |
    | Dimensions (group-by fields) |  |  |  |
    | Filters (+ rationale) |  |  |  |
    | Joins (tables/keys/type) |  |  |  |
    | Aggregation plan |  |  |  |
    | Validation checks |  |  |  |
    | Performance guardrails (optional) |  |  |  |
    
    ### Investigation Plan (tool checklist seeded by Decider)
    
    | Investigation Item | Value (planned tool(s)) | Rationale (why needed) | Comment (stop condition) |
    | --- | --- | --- | --- |
    | Table discovery |  |  |  |
    | Grain confirmation |  |  |  |
    | Column inventory |  |  |  |
    | Time logic / latest date |  |  |  |
    | Status/value distributions |  |  |  |
    | PK / duplicates |  |  |  |
    | Join key / cardinality risk |  |  |  |
    
    ---
    
    ## Example Filled (Ecommerce domain) for query: “all ecommerce products”
    
    > This is what the Decider would fill before tools, using ecommerce_domain.md.
    > 
    
    ### Query Spec (pre-filled by Decider)
    
    | Query Spec Item | Value (from domain.md) | Rationale (from domain.md) | Comment |
    | --- | --- | --- | --- |
    | Business question | “List products in the ecommerce catalog.” | Domain playbook defines “products” as catalog entity. | High confidence. Executor verifies table exists. |
    | Output shape + required columns | Table: `product_id`, `name` (optional: `category`, `price`, `is_active`) | Domain terms map “product” → product catalog fields. | Medium confidence until schema inspected. |
    | Start table + grain | `products` table; grain = 1 row per product/SKU | Domain playbook identifies catalog as dimension-like table. | Must confirm via schema + sample rows. |
    | Time column + default timeframe rule | Time rule: none required for catalog listing; if needed use `updated_at` and latest snapshot | Domain says products are often slowly changing; time optional. | Executor checks if `updated_at` exists; otherwise omit time filter. |
    | Metrics (definitions) | None (pure listing) | Query is a “listing”, not aggregation. | N/A |
    | Dimensions (group-by fields) | None (no group-by) | No aggregation requested. | N/A |
    | Filters (+ rationale) | Default: `is_active = true` (if exists), exclude test/deleted | Domain playbook usually includes “active/valid” filters. | Executor validates whether flags exist; otherwise skip. |
    | Joins (tables/keys/type) | None | Product listing can be satisfied from catalog alone. | If user later asks for sales performance, join to `order_items`. |
    | Aggregation plan | None | No metrics. | N/A |
    | Validation checks | 1) Row count reasonable 2) No duplicate `product_id` in preview | Domain playbook recommends sanity checks to detect wrong grain. | Executor runs quick checks if cheap. |
    | Performance guardrails (optional) | `LIMIT 50` default; avoid `SELECT *` | Domain defaults: safe exploration limits. | Executor enforces LIMIT automatically. |
    
    ### Investigation Plan (tool checklist seeded by Decider)
    
    | Investigation Item | Value (planned tool(s)) | Rationale | Comment (stop condition) |
    | --- | --- | --- | --- |
    | Table discovery | `list_dir(domain_root)` | Confirm `products` exists and locate it. | Stop when `products.*` located. |
    | Grain confirmation | `inspect_table(products)` + preview `LIMIT 10` | Confirm “1 row per product” and identify identifiers. | Stop when `product_id` found and looks unique-ish. |
    | Column inventory | `inspect_table(products)` | Confirm `name`, `category`, `is_active`, `updated_at`. | Stop when required output cols exist. |
    | Time logic / latest date | Only if `updated_at` exists: `max(updated_at)` | Optional; used if domain mandates snapshot behavior. | Skip if no time column or not needed. |
    | Status/value distributions | If `is_active`/`status` exists: `group by status` | Confirm valid filter values. | Skip if no status columns. |
    | PK / duplicates | `count(*)` vs `count(distinct product_id)` (bounded) | Catch duplicates early. | Skip if expensive; do on sample if needed. |
    | Join key / cardinality risk | Not needed (no joins) | Prevent unnecessary joins. | N/A |
    
    ---
    
    ## How this connects to your scratchpad cleanup
    
    This template becomes your **single canonical contract**, replacing “random scratchpad fields”:
    
    - `query_spec` is the checklist
    - `executor_state_pad` is the same checklist **plus evidence** (schemas, samples, distributions, chosen keys)
    - `glossary_facts` feeds the “Metrics/Dimensions/Filters/Joins” rows (with a `source` tag, as you outlined)
    
    ---
    
    If you paste one of your real `domain.md` files (even partial), I’ll fill the same template **exactly** with your domain’s real defaults and vocabulary (no guessing), including the value/rationale/comment for every row.
    
- langgraph nodes
    
    **Short answer — LangGraph/LangChain setup (production-grade, minimal):**
    
    ---
    
    ## 🧠 LangGraph: the control plane (recommended)
    
    ### 1️⃣ Nodes you need (only 5)
    
    ```
    START
      ↓
    DeciderNode
      ↓
    ActionRouter
      ├─ ASK_USER → END
      ├─ BLOCK → END
      └─ EXECUTE → ExecutorGraph
    
    ```
    
    **ExecutorGraph (subgraph):**
    
    ```
    InvestigationNode → SQLNode → EvaluationNode → OutcomeNode
    
    ```
    
    ---
    
    ## 2️⃣ DeciderNode (LLM node)
    
    **LangGraph Node**
    
    - Prompt = **single Decider prompt you wrote**
    - Input = Table 1 fields
    - Output = strict JSON (validated)
    
    ```python
    decider = LLMNode(
        llm=llm,
        prompt=DECIDER_PROMPT,
        output_parser=StrictJSONParser(schema=DeciderSchema)
    )
    
    ```
    
    ---
    
    ## 3️⃣ ActionRouter (pure Python)
    
    **Not an LLM**
    
    ```python
    def route(state):
        match state["decider"]["action"]:
            case "ASK_USER": return "end"
            case "BLOCK": return "end"
            case "EXECUTE": return "executor"
    
    ```
    
    ---
    
    ## 4️⃣ ExecutorGraph (tool-driven, no planning)
    
    ### Nodes inside ExecutorGraph
    
    | Node | Type | Responsibility |
    | --- | --- | --- |
    | InvestigationNode | Python | Run `investigation_plan` tools |
    | SQLNode | LLM | `nl_to_sql_planner` (spec must be complete) |
    | EvaluationNode | Python | `query_result_evaluator` |
    | OutcomeNode | Python | Build Table 8 report |
    
    ---
    
    ## 5️⃣ State object (critical)
    
    **Single shared LangGraph state**
    
    ```python
    class State(TypedDict):
        user_query: str
        domain_md: str
        query_spec: dict
        query_spec_status: dict
        investigation_plan: list
        decider: dict
        executor_report: dict
        attempt_count: int
    
    ```
    
    LangGraph automatically passes this.
    
    ---
    
    ## 6️⃣ Retry loop (LangGraph edge condition)
    
    ```python
    def should_retry(state):
        return (
            state["executor_report"]["status"] == "ERROR"
            and state["attempt_count"] < MAX_ATTEMPTS
        )
    
    ```
    
    Edge:
    
    ```
    OutcomeNode ──retry──▶ DeciderNode
    
    ```
    
    ---
    
    ## 7️⃣ LangChain tools (Executor only)
    
    ```python
    tools = [
      list_dir,
      inspect_table,
      search_glossary,
      execute_sql,
      query_safety_validator
    ]
    
    ```
    
    ⚠️ Decider **never sees tools**, only capability cards in prompt.
    
    ---
    
    ## 8️⃣ Why LangGraph (not plain LangChain)
    
    - Deterministic control flow
    - Explicit state
    - Clean retry edges
    - No hidden planner behavior
    
    ---
    
    ## One-line mental model
    
    > LangGraph = brainstem (control), LangChain = hands (tools), LLM = cortex (decide/specify).
    > 
    
    If you want, I can next:
    
    - Draw the **exact LangGraph diagram**
    - Or give **copy-paste LangGraph code skeleton**
- improvements
    
    Below are the **4 concrete updates** (drop-in, production-ready). They are written as **rules + minimal schemas/pseudocode** so you can implement immediately.
    
    ---
    
    ## 1) Unify the plan field (remove drift)
    
    ### **Change**
    
    Use **one** canonical plan field in the Decider output and in the Executor input:
    
    ✅ Keep: `investigation_plan` (Table 4B)
    
    ❌ Remove (or make alias-only): `decider_packet.tool_sequence`
    
    ### **New rule**
    
    - **Executor executes only `investigation_plan`**.
    - If both exist, `tool_sequence` **must equal** `investigation_plan` or packet is invalid.
    
    ### **Canonical structure**
    
    ```json
    "investigation_plan": [
      {
        "step": 1,
        "tool": "inspect_table",
        "args": {"path": "ecommerce/orders.parquet"},
        "gap_filled": ["start_table", "grain", "time.column"],
        "success_condition": "schema returned with primary identifier and time candidates"
      }
    ]
    
    ```
    
    ---
    
    ## 2) Make “SQL generation gate” code-enforced (not prompt-only)
    
    ### **Change**
    
    Compute a boolean **in Python** before `SQLGenerationNode`:
    
    - `spec_ready_for_sql = True/False`
    
    ### **Required minimum check (authoritative)**
    
    - `business_question`
    - `start_table.path`
    - `grain`
    - `time.column` **OR** explicit `time.rule = "no_time"`
    - `metrics` (can be empty only when output is a pure listing and domain explicitly allows)
    
    ### **Implementation (minimal)**
    
    ```python
    REQUIRED_KEYS = ["business_question", "start_table.path", "grain", "metrics"]
    
    def _get(d, path):
        cur = d
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur: return None
            cur = cur[p]
        return cur
    
    def spec_ready_for_sql(query_spec: dict, query_spec_status: dict) -> tuple[bool, list[str]]:
        missing = []
        # basic required
        for k in REQUIRED_KEYS:
            if _get(query_spec, k) in (None, "", [], {}):
                missing.append(k)
            else:
                st = query_spec_status.get(k.split(".")[0], {}).get("status")
                if st == "missing":
                    missing.append(k)
    
        # time special-case
        time_col = _get(query_spec, "time.column")
        time_rule = _get(query_spec, "time.rule")
        time_status = query_spec_status.get("time", {}).get("status")
    
        if (not time_col) and (time_rule != "no_time"):
            missing.append("time.column")
        elif time_status == "missing":
            missing.append("time")
    
        return (len(missing) == 0, missing)
    
    ```
    
    ### **Executor behavior**
    
    - If `spec_ready_for_sql == False` → return `ERROR` with `failed_checklist_items = missing`
    - SQLGenerationNode **never runs** unless gate passes.
    
    ---
    
    ## 3) Add a “conflict” status to Query Spec Status (for disagreements)
    
    ### **Change**
    
    Extend status taxonomy:
    
    `missing | defaulted | inferred | verified | conflict`
    
    ### **When to set `conflict`**
    
    Executor sets `conflict` when:
    
    - tool results contradict `domain_md` defaults, or
    - tool results contradict prior spec values in a way that changes meaning.
    
    ### **New rule**
    
    - If a **required-minimum** field becomes `conflict` → Executor returns `ERROR` (do not guess).
    
    ### **Example**
    
    ```json
    "time": {
      "status": "conflict",
      "source": "tool_result",
      "notes": "domain_md says order_date, schema only has created_at; requires Decider decision"
    }
    
    ```
    
    ### **Error payload (minimal)**
    
    ```json
    {
      "status": "ERROR",
      "error_type": "SCHEMA",
      "failed_checklist_items": ["time"],
      "minimal_fix_suggestion": "Decider must choose correct time column (created_at vs other) based on domain rules or ask user."
    }
    
    ```
    
    ---
    
    ## 4) Keep the Executor graph linear but support “early halt” safely
    
    ### **Change**
    
    Keep nodes in one line, but allow early exit using a flag:
    
    - `halt_execution: bool`
    
    ### **Rule**
    
    - Any node can set `halt_execution=True` with an `executor_report` (ERROR).
    - All downstream nodes become **no-ops** if `halt_execution=True`.
    
    ### **Minimal pattern**
    
    ```python
    def guard(state):
        return state.get("halt_execution", False)
    
    def investigation_node(state):
        if guard(state): return state
        # run steps; if fail:
        # state["halt_execution"]=True
        # state["executor_report"]=...
        return state
    
    def sql_node(state):
        if guard(state): return state
        ready, missing = spec_ready_for_sql(state["query_spec"], state["query_spec_status"])
        if not ready:
            state["halt_execution"] = True
            state["executor_report"] = {
              "status": "ERROR",
              "error_type": "GRAIN",
              "failed_checklist_items": missing,
              "minimal_fix_suggestion": "Decider must plan steps to close these gaps."
            }
            return state
        # else call nl_to_sql_planner
        return state
    
    ```
    
    ### **Outcome**
    
    - You preserve “no branching” *structurally* (linear graph),
    - but you get correct operational behavior (fast fail, deterministic).
    
    ---
    
    If you implement only **two** things first: do **#1 (plan unification)** and **#2 (SQL gate)**. Those are the biggest production stabilizers.
    

# sprint plan

Here’s a **high-level epic plan** to build the updated, production-ready Decider/Executor system (with the 4 hardening changes).

## Epic 1 — Contracts & Schemas (Source of Truth)

- Finalize the canonical JSON contracts: `DeciderOutput`, `ExecutionPacket`, `ExecutorReport`, `QuerySpec`, `QuerySpecStatus`
- Add `conflict` status to `QuerySpecStatus`
- Remove drift by standardizing on **one** plan field: `investigation_plan` (and deprecate/alias `tool_sequence`)

## Epic 2 — Policy & Gate Enforcement Layer (Hard Guards)

- Implement the **SQL generation gate** (`spec_ready_for_sql`) as a Python predicate
- Implement plan validation (max steps, allowed tools, args schema, success condition required)
- Implement “required minimum” enforcement (including time `no_time` rule)

## Epic 3 — Tooling Layer & Capability Cards Alignment

- Define a strict “tool argument schema” per tool (required args, allowed keys, examples)
- Implement the Executor tool dispatcher (allow-list only, deterministic execution)
- Ensure tool outputs map cleanly into `query_spec` and `query_spec_status` patches

## Epic 4 — Executor Graph Implementation (Linear + Early Halt)

- Build the LangGraph **Executor subgraph** with linear nodes
- Add `halt_execution` mechanism + downstream no-op behavior
- Generate the full **Executor Run Log** and attach minimal snippets to `executor_report` on failures

## Epic 5 — Decider Orchestrator (Compiler Front-End)

- Implement `run_decider()` with strict JSON schema validation + auto-reprompt on schema errors
- Implement “gap-driven planning” logic in the Decider prompt examples/rubric (plan closes only missing/unsafe fields)
- Implement the Action Router: `ASK_USER | EXECUTE | BLOCK`

## Epic 6 — Retry Controller & Minimal Replan Loop

- Implement controller loop: Decider → Executor → (if ERROR) Decider with `last_executor_report`
- Enforce `policy_limits.max_attempts`
- Ensure replans are minimal: only address `failed_checklist_items`

## Epic 7 — Observability, Auditability, and Developer UX

- Structured tracing for every tool call (inputs, outputs, duration, errors)
- Deterministic run logs persisted per attempt (Tables 7/8 artifacts)
- Debug views / exportable JSON for replaying runs in tests

## Epic 8 — Test Harness & Production Readiness

- Unit tests for schema validation, SQL gate, conflict detection, and tool arg validation
- Golden-test fixtures: known queries → expected spec/status → expected investigation_plan
- Failure-mode tests: missing fields, schema mismatch, empty results, policy blocks, timeouts

If you want next, I can turn these epics into a **1–2 sprint phased delivery** (still high level) or map each epic to your existing repo modules (`planner_state.py`, prompt registry, tool classes, LangGraph wiring).

Below is the **complete, aligned “gap-closure pack”**: the **two missing prompts** + the **fixed strict JSON schemas** + the **canonical investigation plan schema**.
All names/fields match the plan and tables we already agreed on. No drift.

---

# A) Missing Prompts (Aligned)

## A1) `nl_to_sql_planner.md` (SQL-only output)

```md
ROLE
You are the SQL Generator.

Your job is to convert a completed Query Spec into SQL.

HARD CONSTRAINTS
- Output SQL ONLY. No prose, no markdown.
- You MUST NOT invent tables, columns, joins, metrics, or filters.
- You MUST NOT guess missing fields.
- You MUST NOT modify the Query Spec.
- If the Query Spec is incomplete for SQL generation, output exactly:
  ERROR: QUERY_SPEC_INCOMPLETE

INPUTS
- query_spec (conforms to query_spec.schema.json)
- query_spec_status (conforms to query_spec_status.schema.json)
- schema_snapshot (optional; may include table columns/types)

REQUIRED PRECONDITIONS (must be satisfied or FAIL)
The following must be present and not missing/conflict in query_spec_status:
- business_question
- start_table_grain
- time (unless query_spec.time.rule == "no_time")
- metrics

SQL RULES
- Use only query_spec.start_table.path as the base FROM source.
- Use only query_spec.joins when additional tables are needed.
- Apply query_spec.filters exactly.
- Produce columns matching query_spec.output_shape.columns when provided (otherwise return sensible minimal columns implied by grain/metrics).
- Implement query_spec.aggregation_plan exactly and preserve query_spec.grain.
- Apply query_spec.time.rule and query_spec.time.n_days when applicable (unless "no_time").
- Respect query_spec.performance_guardrails (e.g., LIMIT, avoid SELECT *).

OUTPUT
Return a single SQL string.
```

---

## A2) `query_result_evaluator.md` (JSON-only output)

```md
ROLE
You are the Result Evaluator.

Your job is to determine whether the query results satisfy the Query Spec.

HARD CONSTRAINTS
- Output JSON ONLY (no prose).
- You MUST NOT reinterpret or change the Query Spec.
- You MUST NOT propose SQL changes (only describe issues).
- You MUST NOT invent expectations not present in query_spec.

INPUTS
- query_spec
- results_summary (may include row_count, column_names, sample_rows, aggregate_totals)
- validation_checks (from query_spec.validation_checks)

EVALUATION STEPS
1) Output shape check:
   - If query_spec.output_shape.columns is non-empty, verify those columns exist in results_summary.column_names.
2) Grain check:
   - Verify results are consistent with query_spec.grain (using row_count, distinct counts if provided).
3) Sanity checks:
   - Flag zero rows if business_question implies results should exist (only if validation_checks mention it).
4) Apply validation_checks exactly as written (interpret as pass/fail checks).

OUTPUT (JSON)
{
  "satisfied": true | false,
  "issues": [string],
  "notes": string
}
```

---

# B) Fixed JSON Schemas (Strict + Fully Aligned)

## B1) `query_spec.schema.json` (Table 9)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "query_spec.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "business_question",
    "output_shape",
    "start_table",
    "grain",
    "time",
    "metrics",
    "dimensions",
    "filters",
    "joins",
    "aggregation_plan",
    "validation_checks",
    "performance_guardrails",
    "defaults_used",
    "open_questions"
  ],
  "properties": {
    "business_question": { "type": "string" },

    "output_shape": {
      "type": "object",
      "additionalProperties": false,
      "required": ["type", "columns"],
      "properties": {
        "type": { "type": "string" },
        "columns": { "type": "array", "items": { "type": "string" } }
      }
    },

    "start_table": {
      "type": "object",
      "additionalProperties": false,
      "required": ["name", "path"],
      "properties": {
        "name": { "type": "string" },
        "path": { "type": "string" }
      }
    },

    "grain": { "type": "string" },

    "time": {
      "type": "object",
      "additionalProperties": false,
      "required": ["column", "rule", "n_days"],
      "properties": {
        "column": { "type": "string" },
        "rule": { "type": "string" },
        "n_days": { "type": ["integer", "null"] }
      }
    },

    "metrics": { "type": "array" },
    "dimensions": { "type": "array" },
    "filters": { "type": "array" },
    "joins": { "type": "array" },

    "aggregation_plan": { "type": "string" },
    "validation_checks": { "type": "array" },
    "performance_guardrails": { "type": "array" },
    "defaults_used": { "type": "array" },
    "open_questions": { "type": "array" }
  }
}
```

---

## B2) `query_spec_status.schema.json` (Table 10)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "query_spec_status.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "business_question",
    "output_shape",
    "start_table_grain",
    "time",
    "metrics",
    "dimensions",
    "filters",
    "joins",
    "aggregation_plan",
    "validation_checks",
    "performance_guardrails"
  ],
  "$defs": {
    "status_item": {
      "type": "object",
      "additionalProperties": false,
      "required": ["status", "source", "notes", "blocks_execution"],
      "properties": {
        "status": { "enum": ["missing", "defaulted", "inferred", "verified", "conflict"] },
        "source": { "enum": ["domain_md", "tool_result", "user", "rule"] },
        "notes": { "type": "string" },
        "blocks_execution": { "type": "boolean" }
      }
    }
  },
  "properties": {
    "business_question": { "$ref": "#/$defs/status_item" },
    "output_shape": { "$ref": "#/$defs/status_item" },
    "start_table_grain": { "$ref": "#/$defs/status_item" },
    "time": { "$ref": "#/$defs/status_item" },
    "metrics": { "$ref": "#/$defs/status_item" },
    "dimensions": { "$ref": "#/$defs/status_item" },
    "filters": { "$ref": "#/$defs/status_item" },
    "joins": { "$ref": "#/$defs/status_item" },
    "aggregation_plan": { "$ref": "#/$defs/status_item" },
    "validation_checks": { "$ref": "#/$defs/status_item" },
    "performance_guardrails": { "$ref": "#/$defs/status_item" }
  }
}
```

---

## B3) `investigation_plan_step.schema.json` (Table 4B canonical)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "investigation_plan_step.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["step", "tool", "args", "fills_gap", "success_condition"],
  "properties": {
    "step": { "type": "integer", "minimum": 1 },
    "tool": {
      "enum": [
        "list_dir",
        "inspect_table",
        "preview_rows",
        "search_glossary",
        "nl_to_sql_planner",
        "sql_plan_updater",
        "query_safety_validator",
        "execute_sql",
        "query_result_evaluator"
      ]
    },
    "args": { "type": "object" },
    "fills_gap": { "type": "string" },
    "success_condition": { "type": "string" }
  }
}
```

---

## B4) `executor_report.schema.json` (Table 8, conditional branches)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "executor_report.schema.json",
  "oneOf": [
    {
      "title": "SUCCESS",
      "type": "object",
      "additionalProperties": false,
      "required": ["status", "final_sql", "result_summary", "evaluation", "finished_output"],
      "properties": {
        "status": { "const": "SUCCESS" },
        "final_sql": { "type": "string" },
        "result_summary": { "type": "string" },
        "evaluation": { "type": "object" },
        "finished_output": { "type": "string" }
      }
    },
    {
      "title": "ERROR",
      "type": "object",
      "additionalProperties": false,
      "required": ["status", "error_type", "failed_checklist_items", "what_changed", "minimal_fix_suggestion", "last_sql", "last_error"],
      "properties": {
        "status": { "const": "ERROR" },
        "error_type": { "enum": ["SCHEMA", "SQL", "EMPTY", "GRAIN", "POLICY", "NO_DATASET", "AMBIGUOUS"] },
        "failed_checklist_items": { "type": "array", "items": { "type": "string" } },
        "what_changed": { "type": "string" },
        "minimal_fix_suggestion": { "type": "string" },
        "last_sql": { "type": "string" },
        "last_error": { "type": "string" }
      }
    }
  ]
}
```

---

## B5) `policy_limits.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "policy_limits.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["max_attempts", "max_rows", "timeout_seconds", "allow_cross_join"],
  "properties": {
    "max_attempts": { "type": "integer", "minimum": 1 },
    "max_rows": { "type": "integer", "minimum": 1 },
    "timeout_seconds": { "type": "integer", "minimum": 1 },
    "allow_cross_join": { "type": "boolean" }
  }
}
```

---

## B6) `decider_output.schema.json` (Fully strict + references)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "decider_output.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "action",
    "domain",
    "intent",
    "decisions",
    "query_spec",
    "query_spec_status",
    "investigation_plan",
    "expected_output",
    "stop_conditions",
    "ask_user",
    "block_reason"
  ],
  "properties": {
    "action": { "enum": ["ASK_USER", "EXECUTE", "BLOCK"] },
    "domain": { "type": "string" },
    "intent": { "type": "string" },

    "decisions": {
      "type": "object",
      "additionalProperties": false,
      "required": ["comprehension", "determinacy", "clarification_need"],
      "properties": {
        "comprehension": { "enum": ["INTELLIGIBLE", "UNINTELLIGIBLE"] },
        "determinacy": { "enum": ["DETERMINED", "UNDERDETERMINED"] },
        "clarification_need": { "enum": ["ASK_REQUIRED", "DEFAULT_OK"] }
      }
    },

    "query_spec": { "$ref": "query_spec.schema.json" },
    "query_spec_status": { "$ref": "query_spec_status.schema.json" },

    "investigation_plan": {
      "type": "array",
      "maxItems": 4,
      "items": { "$ref": "investigation_plan_step.schema.json" }
    },

    "expected_output": { "type": "string" },
    "stop_conditions": { "type": "array", "items": { "type": "string" } },

    "ask_user": {
      "type": "object",
      "additionalProperties": false,
      "required": ["question", "why_non_defaultable", "what_answer_unblocks"],
      "properties": {
        "question": { "type": "string" },
        "why_non_defaultable": { "type": "string" },
        "what_answer_unblocks": { "type": "string" }
      }
    },

    "block_reason": { "type": "string" }
  }
}
```

---

# C) Canonical naming reminders (to avoid drift)

* Investigation plan step keys are **exactly**: `step`, `tool`, `args`, `fills_gap`, `success_condition`
* Query Spec Status keys are **exactly**:
  `business_question`, `output_shape`, `start_table_grain`, `time`, `metrics`, `dimensions`, `filters`, `joins`, `aggregation_plan`, `validation_checks`, `performance_guardrails`

---

If you want the *next* missing piece (still aligned): I can provide **tool argument schemas** for each allowed tool (list_dir, inspect_table, etc.) in the same strict JSON Schema style.


Yes. Here are the **missing P1 items**, **fully aligned** with the current plan, tables, prompts, and the strict schemas (B1–B6). No behavior changes.

---

## P1-A) `domain.md` template (canonical)

```md
# Domain: <domain_name>

## 1) Domain identity
- domain_key: <ecommerce|mr|ccr|generic>
- description: <short>

## 2) Time semantics (Decider reference)
- default_time_column: <string or empty>
- default_time_rule: <e.g., "last_n_days" | "as_of_latest" | "no_time">
- default_time_n_days: <int or empty>
- supports_no_time_queries: <true|false>

## 3) Listing rules
- listing_allows_empty_metrics: <true|false>
- listing_default_limit: <int>

## 4) Core entities (optional hints)
- primary_entities:
  - name: <entity>
    typical_grain: <"one row per ...">
    default_start_table_hint: <table logical name>

## 5) Metric dictionary (Decider reference; Executor may verify via glossary/schema)
- metrics:
  - metric_name: <string>
    definition: <string>
    default_filters: [<string>]
    required_tables: [<string>]
    disallowed_grains: [<string>]

## 6) Join conventions (optional hints)
- canonical_joins:
  - left_table: <string>
    right_table: <string>
    on: <string>
    join_type: <string>
- forbidden_joins: [<string>]

## 7) Safety defaults (Executor reference)
- performance_guardrails:
  - default_limit: <int>
  - avoid_select_star: <true|false>
  - allow_cross_join: <true|false>

## 8) Notes
- notes: <free text>
```

---

## P1-B) Example `domain.md` (ecommerce)

```md
# Domain: ecommerce

## 1) Domain identity
- domain_key: ecommerce
- description: ECommerce warehouse covering customers, orders, order_items, products, payments, inventory.

## 2) Time semantics (Decider reference)
- default_time_column: created_at
- default_time_rule: last_n_days
- default_time_n_days: 30
- supports_no_time_queries: true

## 3) Listing rules
- listing_allows_empty_metrics: true
- listing_default_limit: 50

## 4) Core entities (optional hints)
- primary_entities:
  - name: orders
    typical_grain: one row per order
    default_start_table_hint: orders
  - name: customers
    typical_grain: one row per customer
    default_start_table_hint: customers
  - name: products
    typical_grain: one row per product
    default_start_table_hint: products

## 5) Metric dictionary (Decider reference; Executor may verify via glossary/schema)
- metrics:
  - metric_name: revenue
    definition: Total paid amount for completed/settled payments.
    default_filters: ["payment_status in ('settled','paid')"]
    required_tables: ["payments"]
    disallowed_grains: ["one row per customer_session"]
  - metric_name: order_count
    definition: Count of distinct orders.
    default_filters: []
    required_tables: ["orders"]
    disallowed_grains: []

## 6) Join conventions (optional hints)
- canonical_joins:
  - left_table: orders
    right_table: order_items
    on: orders.order_id = order_items.order_id
    join_type: left
  - left_table: orders
    right_table: customers
    on: orders.customer_id = customers.customer_id
    join_type: left
- forbidden_joins: ["products x customers (no key)"]

## 7) Safety defaults (Executor reference)
- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false

## 8) Notes
- notes: Use created_at as default time axis unless query explicitly requests a different business date.
```

This supports the rule you flagged: **`listing_allows_empty_metrics: true`** (used only by the **SQL gate** and only when the Decider marks it as such in `query_spec.output_shape.type` / query intent).

---

## P1-C) Decider prompt examples (full JSON, schema-valid)

### Example 1 — `ASK_USER` (non-defaultable ambiguity)

```json
{
  "action": "ASK_USER",
  "domain": "ecommerce",
  "intent": "NEW_QUERY",
  "decisions": {
    "comprehension": "INTELLIGIBLE",
    "determinacy": "UNDERDETERMINED",
    "clarification_need": "ASK_REQUIRED"
  },
  "query_spec": {
    "business_question": "Get revenue for a specific period.",
    "output_shape": { "type": "table", "columns": ["revenue"] },
    "start_table": { "name": "", "path": "" },
    "grain": "",
    "time": { "column": "created_at", "rule": "last_n_days", "n_days": 30 },
    "metrics": ["revenue"],
    "dimensions": [],
    "filters": [],
    "joins": [],
    "aggregation_plan": "Aggregate revenue over the selected period.",
    "validation_checks": ["revenue is non-negative", "row_count is 1"],
    "performance_guardrails": ["limit 1", "avoid_select_star"],
    "defaults_used": ["time.rule=last_n_days", "time.n_days=30"],
    "open_questions": ["Which entity scope should revenue be computed for?"]
  },
  "query_spec_status": {
    "business_question": { "status": "verified", "source": "user", "notes": "User asked for revenue.", "blocks_execution": false },
    "output_shape": { "status": "inferred", "source": "rule", "notes": "Revenue output as single-column table.", "blocks_execution": false },
    "start_table_grain": { "status": "missing", "source": "rule", "notes": "Need to know whether revenue is order-level, payment-level, or customer-level for correct start table and grain.", "blocks_execution": true },
    "time": { "status": "defaulted", "source": "domain_md", "notes": "Default time semantics applied.", "blocks_execution": true },
    "metrics": { "status": "verified", "source": "user", "notes": "Metric requested: revenue.", "blocks_execution": true },
    "dimensions": { "status": "inferred", "source": "rule", "notes": "No grouping requested.", "blocks_execution": false },
    "filters": { "status": "inferred", "source": "rule", "notes": "No explicit filters provided.", "blocks_execution": false },
    "joins": { "status": "missing", "source": "rule", "notes": "Join requirements depend on start table selection.", "blocks_execution": true },
    "aggregation_plan": { "status": "inferred", "source": "rule", "notes": "Aggregation needed for revenue.", "blocks_execution": true },
    "validation_checks": { "status": "defaulted", "source": "rule", "notes": "Default sanity checks.", "blocks_execution": false },
    "performance_guardrails": { "status": "defaulted", "source": "domain_md", "notes": "Default safety guardrails.", "blocks_execution": false }
  },
  "investigation_plan": [],
  "expected_output": "A single revenue value for the requested scope/time period.",
  "stop_conditions": ["Do not execute until start_table_grain and joins are resolved."],
  "ask_user": {
    "question": "When you say revenue, do you mean payments received (paid/settled), or order totals (placed orders)?",
    "why_non_defaultable": "Different definitions change the start table, filters, and the computed result.",
    "what_answer_unblocks": "Selecting the revenue definition determines the correct start table, grain, and required joins/filters."
  },
  "block_reason": ""
}
```

---

### Example 2 — `EXECUTE` (packet ready; Decider sends plan)

```json
{
  "action": "EXECUTE",
  "domain": "ecommerce",
  "intent": "NEW_QUERY",
  "decisions": {
    "comprehension": "INTELLIGIBLE",
    "determinacy": "DETERMINED",
    "clarification_need": "DEFAULT_OK"
  },
  "query_spec": {
    "business_question": "List products in the ecommerce catalog.",
    "output_shape": { "type": "table", "columns": ["product_id", "name"] },
    "start_table": { "name": "products", "path": "ecommerce/products" },
    "grain": "one row per product",
    "time": { "column": "", "rule": "no_time", "n_days": null },
    "metrics": [],
    "dimensions": [],
    "filters": [],
    "joins": [],
    "aggregation_plan": "No aggregation (listing).",
    "validation_checks": ["row_count > 0", "no duplicate product_id in sample"],
    "performance_guardrails": ["limit 50", "avoid_select_star"],
    "defaults_used": ["limit=50"],
    "open_questions": []
  },
  "query_spec_status": {
    "business_question": { "status": "verified", "source": "user", "notes": "User asked for product listing.", "blocks_execution": false },
    "output_shape": { "status": "inferred", "source": "rule", "notes": "Listing output as table with id + name.", "blocks_execution": false },
    "start_table_grain": { "status": "inferred", "source": "domain_md", "notes": "Products table is the default catalog start table; executor will verify schema and grain.", "blocks_execution": true },
    "time": { "status": "verified", "source": "domain_md", "notes": "Domain supports no_time queries for catalog listing.", "blocks_execution": false },
    "metrics": { "status": "verified", "source": "domain_md", "notes": "Listing allows empty metrics.", "blocks_execution": false },
    "dimensions": { "status": "verified", "source": "rule", "notes": "No grouping.", "blocks_execution": false },
    "filters": { "status": "verified", "source": "rule", "notes": "No explicit filters.", "blocks_execution": false },
    "joins": { "status": "verified", "source": "rule", "notes": "No joins needed for listing.", "blocks_execution": false },
    "aggregation_plan": { "status": "verified", "source": "rule", "notes": "No aggregation required.", "blocks_execution": false },
    "validation_checks": { "status": "defaulted", "source": "rule", "notes": "Default listing sanity checks.", "blocks_execution": false },
    "performance_guardrails": { "status": "defaulted", "source": "domain_md", "notes": "Default safe limit.", "blocks_execution": false }
  },
  "investigation_plan": [
    {
      "step": 1,
      "tool": "inspect_table",
      "args": { "path": "ecommerce/products" },
      "fills_gap": "start_table_grain",
      "success_condition": "Schema includes product_id and name; confirm row meaning supports one row per product."
    }
  ],
  "expected_output": "A table of products with product_id and name (limited to 50 rows).",
  "stop_conditions": ["If schema does not contain required columns, return ERROR with failed_checklist_items."],
  "ask_user": {
    "question": "",
    "why_non_defaultable": "",
    "what_answer_unblocks": ""
  },
  "block_reason": ""
}
```

These examples are **strictly valid** under `decider_output.schema.json` + Table 9/10 schemas.

--
Below is the **single copy-paste pack** with **everything from my previous response**, updated to **include `preview_rows` consistently** and **include `AMBIGUOUS`** in `executor_report.schema.json`.
(Only these two optional polishes were applied; nothing else changed.)

---

# 1) Tool Argument + Output Schemas (Allowed Tools)

## 1.1 `list_dir.tool.schema.json`

```json
{
  "$id": "list_dir.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path"],
      "properties": { "path": { "type": "string" } }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["entries"],
      "properties": {
        "entries": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "path", "type"],
            "properties": {
              "name": { "type": "string" },
              "path": { "type": "string" },
              "type": { "enum": ["file", "dir"] }
            }
          }
        }
      }
    }
  }
}
```

---

## 1.2 `inspect_table.tool.schema.json`

```json
{
  "$id": "inspect_table.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path"],
      "properties": { "path": { "type": "string" } }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["columns"],
      "properties": {
        "columns": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "type"],
            "properties": {
              "name": { "type": "string" },
              "type": { "type": "string" }
            }
          }
        },
        "row_count_estimate": { "type": ["integer", "null"] },
        "primary_key_candidates": { "type": "array", "items": { "type": "string" } },
        "time_column_candidates": { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

---

## 1.3 `preview_rows.tool.schema.json`

```json
{
  "$id": "preview_rows.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "limit"],
      "properties": {
        "path": { "type": "string" },
        "limit": { "type": "integer", "minimum": 1, "maximum": 100 }
      }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["rows_preview"],
      "properties": {
        "rows_preview": { "type": "array", "items": { "type": "object" } }
      }
    }
  }
}
```

---

## 1.4 `search_glossary.tool.schema.json`

```json
{
  "$id": "search_glossary.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["term", "domain"],
      "properties": {
        "term": { "type": "string" },
        "domain": { "type": "string" }
      }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["hits"],
      "properties": {
        "hits": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["title", "snippet", "source_ref"],
            "properties": {
              "title": { "type": "string" },
              "snippet": { "type": "string" },
              "source_ref": { "type": "string" }
            }
          }
        }
      }
    }
  }
}
```

---

## 1.5 `execute_sql.tool.schema.json`

```json
{
  "$id": "execute_sql.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["sql"],
      "properties": {
        "sql": { "type": "string" },
        "timeout_seconds": { "type": ["integer", "null"] },
        "max_rows": { "type": ["integer", "null"] }
      }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["row_count", "columns", "rows_preview"],
      "properties": {
        "row_count": { "type": "integer" },
        "columns": { "type": "array", "items": { "type": "string" } },
        "rows_preview": { "type": "array", "items": { "type": "object" } }
      }
    }
  }
}
```

---

## 1.6 `query_safety_validator.tool.schema.json`

```json
{
  "$id": "query_safety_validator.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["sql", "policy_limits"],
      "properties": {
        "sql": { "type": "string" },
        "policy_limits": { "$ref": "policy_limits.schema.json" }
      }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["allowed", "flags"],
      "properties": {
        "allowed": { "type": "boolean" },
        "flags": { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

---

## 1.7 `sql_plan_updater.tool.schema.json`

```json
{
  "$id": "sql_plan_updater.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["sql", "patch_instructions"],
      "properties": {
        "sql": { "type": "string" },
        "patch_instructions": { "type": "string" }
      }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["sql"],
      "properties": { "sql": { "type": "string" } }
    }
  }
}
```

---

## 1.8 `nl_to_sql_planner.tool.schema.json`

```json
{
  "$id": "nl_to_sql_planner.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["query_spec", "query_spec_status"],
      "properties": {
        "query_spec": { "$ref": "query_spec.schema.json" },
        "query_spec_status": { "$ref": "query_spec_status.schema.json" }
      }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["sql"],
      "properties": { "sql": { "type": "string" } }
    }
  }
}
```

---

## 1.9 `query_result_evaluator.tool.schema.json`

```json
{
  "$id": "query_result_evaluator.tool.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["args", "returns"],
  "properties": {
    "args": {
      "type": "object",
      "additionalProperties": false,
      "required": ["query_spec", "results_summary", "validation_checks"],
      "properties": {
        "query_spec": { "$ref": "query_spec.schema.json" },
        "results_summary": { "type": "object" },
        "validation_checks": { "type": "array" }
      }
    },
    "returns": {
      "type": "object",
      "additionalProperties": false,
      "required": ["satisfied", "issues", "notes"],
      "properties": {
        "satisfied": { "type": "boolean" },
        "issues": { "type": "array", "items": { "type": "string" } },
        "notes": { "type": "string" }
      }
    }
  }
}
```

---

# 2) Updated `investigation_plan_step.schema.json` (now includes `preview_rows`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "investigation_plan_step.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["step", "tool", "args", "fills_gap", "success_condition"],
  "properties": {
    "step": { "type": "integer", "minimum": 1 },
    "tool": {
      "enum": [
        "list_dir",
        "inspect_table",
        "preview_rows",
        "search_glossary",
        "nl_to_sql_planner",
        "sql_plan_updater",
        "query_safety_validator",
        "execute_sql",
        "query_result_evaluator"
      ]
    },
    "args": { "type": "object" },
    "fills_gap": { "type": "string" },
    "success_condition": { "type": "string" }
  }
}
```

---

# 3) Updated `executor_report.schema.json` (now includes `AMBIGUOUS`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "executor_report.schema.json",
  "oneOf": [
    {
      "title": "SUCCESS",
      "type": "object",
      "additionalProperties": false,
      "required": ["status", "final_sql", "result_summary", "evaluation", "finished_output"],
      "properties": {
        "status": { "const": "SUCCESS" },
        "final_sql": { "type": "string" },
        "result_summary": { "type": "string" },
        "evaluation": { "type": "object" },
        "finished_output": { "type": "string" }
      }
    },
    {
      "title": "ERROR",
      "type": "object",
      "additionalProperties": false,
      "required": ["status", "error_type", "failed_checklist_items", "what_changed", "minimal_fix_suggestion", "last_sql", "last_error"],
      "properties": {
        "status": { "const": "ERROR" },
        "error_type": { "enum": ["SCHEMA", "SQL", "EMPTY", "GRAIN", "POLICY", "NO_DATASET", "AMBIGUOUS"] },
        "failed_checklist_items": { "type": "array", "items": { "type": "string" } },
        "what_changed": { "type": "string" },
        "minimal_fix_suggestion": { "type": "string" },
        "last_sql": { "type": "string" },
        "last_error": { "type": "string" }
      }
    }
  ]
}
```

---

# 4) SQL gate domain check (Markdown key-value scan; no YAML)

```python
import re

def get_domain_bool(domain_md: str, key: str) -> bool | None:
    # matches: "- listing_allows_empty_metrics: true"
    m = re.search(rf"(?im)^\s*-\s*{re.escape(key)}\s*:\s*(true|false)\s*$", domain_md)
    if not m:
        return None
    return m.group(1).lower() == "true"

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

---

# 5) Updated Table 6 row (Executor Allowed Tools) — only the new tool

Add this row to Table 6:

* `preview_rows | ✅ | Sample rows for grain/filter intuition | Optional; limited to <=100 rows`

---

If you want, I can also produce the **final combined `/instructions` folder layout** (filenames + where each schema/prompt goes) so your team can drop it into the repo with zero ambiguity.

