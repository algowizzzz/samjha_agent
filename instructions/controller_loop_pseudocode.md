Here is the **controller loop pseudocode skeleton** (drop-in for `parquet_agent.py`). It is **fully aligned** with our contracts:

* Decider: **one prompt call**, emits `decider_output` (schema-valid)
* Executor: runs **linear graph** once, emits `executor_report` (schema-valid)
* Retry: controller re-calls Decider with `last_executor_report`
* `attempt_count` lives in backend state (and can be mirrored in scratchpad if you want, but not required)
* `max_attempts` enforced **only** by controller

```python
# parquet_agent.py (pseudocode skeleton)

from typing import TypedDict, Optional, Dict, Any

# --- State shape (backend/controller-owned) ---

class ControllerState(TypedDict):
    user_query: str
    conversation_history: list

    domain_md: str  # selected domain content (string)
    policy_limits: dict  # must include max_attempts, max_rows, timeout_seconds, allow_cross_join

    # canonical contract
    query_spec: dict
    query_spec_status: dict

    # retry context
    last_executor_report: Optional[dict]
    attempt_count: int  # controller-owned counter


# --- Hooks you already have / will implement ---

def load_domain_md(user_query: str, conversation_history: list) -> str:
    """Return domain_md text. (Selection logic can be simple or tool-based elsewhere.)"""
    ...

def run_decider(state: ControllerState) -> dict:
    """
    Single LLM call using Decider prompt.
    Must validate output against decider_output.schema.json.
    If schema invalid, re-prompt internally until valid or fail (controller policy).
    """
    ...

def run_executor(decider_output: dict, state: ControllerState) -> dict:
    """
    Runs Executor subgraph exactly once (Investigation -> SQL -> Safety -> Execute -> Evaluate -> Outcome).
    Must validate output against executor_report.schema.json.
    """
    ...

def render_ask_user(decider_output: dict) -> dict:
    """Return response to UI: question + context fields."""
    return {
        "status": "ASK_USER",
        "question": decider_output["ask_user"]["question"],
        "why_non_defaultable": decider_output["ask_user"]["why_non_defaultable"],
        "what_answer_unblocks": decider_output["ask_user"]["what_answer_unblocks"]
    }

def render_block(decider_output: dict) -> dict:
    """Return blocking response to UI."""
    return {"status": "BLOCK", "reason": decider_output.get("block_reason", "Blocked.")}

def render_success(executor_report: dict) -> dict:
    """Return final user-facing output."""
    return {
        "status": "SUCCESS",
        "finished_output": executor_report["finished_output"],
        "final_sql": executor_report["final_sql"],
        "result_summary": executor_report["result_summary"]
    }

def render_error_max_attempts(last_report: dict, attempt_count: int, max_attempts: int) -> dict:
    return {
        "status": "ERROR",
        "reason": "Max attempts reached.",
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "last_executor_report": last_report
    }


# --- Controller loop (the missing piece) ---

def handle_query(user_query: str, conversation_history: list, prior_state: Optional[ControllerState] = None) -> dict:
    """
    Controller orchestrates:
      Decider -> Executor -> (ERROR -> Decider retry) until SUCCESS / ASK_USER / BLOCK / max_attempts.
    """

    # 1) Initialize controller state
    state: ControllerState = prior_state or {
        "user_query": user_query,
        "conversation_history": conversation_history,

        "domain_md": load_domain_md(user_query, conversation_history),
        "policy_limits": {
            # policy_limits should come from config; shown inline for pseudocode
            "max_attempts": 3,
            "max_rows": 5000,
            "timeout_seconds": 30,
            "allow_cross_join": False
        },

        "query_spec": {},
        "query_spec_status": {},

        "last_executor_report": None,
        "attempt_count": 0
    }

    max_attempts = int(state["policy_limits"]["max_attempts"])

    # 2) Main loop — controller-owned retries only
    while True:
        # Enforce max_attempts BEFORE calling executor (attempt_count counts executor runs)
        if state["attempt_count"] >= max_attempts and state["last_executor_report"] is not None:
            return render_error_max_attempts(state["last_executor_report"], state["attempt_count"], max_attempts)

        # 2A) Call Decider (one prompt call)
        decider_output = run_decider(state)  # schema-valid decider_output

        action = decider_output["action"]

        # Keep canonical spec in controller state (single contract)
        state["query_spec"] = decider_output["query_spec"]
        state["query_spec_status"] = decider_output["query_spec_status"]

        # 2B) Route actions
        if action == "ASK_USER":
            # Controller returns question to UI; loop stops until user replies with more info
            return render_ask_user(decider_output)

        if action == "BLOCK":
            return render_block(decider_output)

        if action != "EXECUTE":
            # Defensive: schema should prevent this, but keep safe
            return {"status": "ERROR", "reason": f"Invalid action from Decider: {action}"}

        # 2C) Execute once (increments attempt_count)
        state["attempt_count"] += 1

        executor_report = run_executor(decider_output, state)  # schema-valid executor_report
        state["last_executor_report"] = executor_report

        # 2D) Interpret executor outcome
        if executor_report["status"] == "SUCCESS":
            return render_success(executor_report)

        # executor_report["status"] == "ERROR" => controller loops back to Decider
        # Decider will see last_executor_report and produce a minimal revised plan or ASK_USER/BLOCK.
        continue
```

### Key points (answers to your 3 questions)

* **Where does `attempt_count` live?**
  In **ControllerState**, controller-owned (`parquet_agent.py`). It increments **once per executor run**.

* **How is `max_attempts` enforced?**
  In the controller loop, **before starting another executor run**.

* **How does Decider → Executor → Decider retry happen?**
  Controller stores `last_executor_report` into state and simply loops. Next `run_decider(state)` sees it and produces a minimal replan (or ASK_USER/BLOCK).

If you want, I can also add the **state persistence hooks** (`load_state(session_id) / save_state(session_id)`) to make retries survive process restarts—still aligned with the same loop.
