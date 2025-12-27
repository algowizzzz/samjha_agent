# QUERY_SPEC_INCOMPLETE Error Explanation

## Error Message
```
Data file not found: data/sqlselect/sales.csv
SQL generation error: ERROR: QUERY_SPEC_INCOMPLETE
```

## Root Cause

This error occurs when the **Decider** creates a Query Spec with **blocking fields** that remain **missing** or in **conflict** after investigation, and the **SQL Generator** detects these gaps before attempting SQL generation.

---

## Decider Prompt Rules That Lead to This Error

### 1. REQUIRED MINIMUM FOR EXECUTION (Lines 115-122)

The Decider prompt requires these fields to be **not missing**:

```115:122:external/config/prompts/decider.md
## REQUIRED MINIMUM FOR EXECUTION

All must be **not missing**:
- `business_question`
- `start_table.path`
- `grain`
- `metrics`
- `time.column` **OR** `time.rule = "no_time"`

Verification happens in the Executor.
```

**Impact**: If `start_table.path` is missing, the Decider must either:
- Add an investigation plan to find it (if tool-resolvable)
- Ask the user (if not tool-resolvable)
- Block the query (if cannot be resolved)

---

### 2. Blocking Execution Fields (Line 236)

The Query Spec Status schema defines which fields **block execution**:

```236:236:external/config/prompts/decider.md
  "start_table_grain": { "status": "", "source": "", "notes": "", "blocks_execution": true },
```

**Impact**: `start_table_grain` has `blocks_execution: true`, meaning:
- If its status is `"missing"` or `"conflict"`, SQL generation cannot proceed
- The Decider must resolve this before allowing execution

---

### 3. Investigation Plan Trigger (Lines 86-91)

The Decider must create investigation steps for blocking gaps:

```86:91:external/config/prompts/decider.md
**Trigger A: Blocking Fields**
For every **blocking** field (`blocks_execution = true`) that is:
- `missing`, `inferred`, or `conflict`
- **and tool-resolvable**

➡️ Add investigation steps to resolve it
```

**Impact**: When `start_table.path` is missing:
- Decider adds `list_dir` investigation step to find the file
- Sets action to `EXECUTE` to let Executor run investigation
- If investigation fails (file not found), the gap remains `"missing"`

---

### 4. SQL Generator Check (nl_to_sql_planner.py)

The SQL generator checks for blocking gaps before generating SQL:

```100:103:external/tools/parquet_agent/nl_to_sql_planner.py
        # Check for blocking gaps
        for field, status in query_spec_status.items():
            if status.get("blocks_execution") and status.get("status") in ["missing", "conflict"]:
                return {"sql": "ERROR: QUERY_SPEC_INCOMPLETE"}
```

**Impact**: If any field with `blocks_execution: true` has status `"missing"` or `"conflict"`, SQL generation is blocked and returns the error.

---

## Error Flow

1. **User Query**: User asks a question that requires a table (e.g., "show me sales data")

2. **Decider Analysis**:
   - Decider identifies `start_table.path` is missing
   - Sets `start_table_grain.status = "missing"` with `blocks_execution: true`
   - Adds investigation plan: `list_dir` to find the file
   - Sets action: `EXECUTE`

3. **Executor Investigation**:
   - Runs `list_dir` to find `sales.csv`
   - File not found → investigation fails
   - `start_table_grain.status` remains `"missing"`

4. **SQL Generation Attempt**:
   - `sql_generation_node` checks `spec_ready_for_sql()`
   - Or `nl_to_sql_planner` checks for blocking gaps
   - Finds `start_table_grain` with `status: "missing"` and `blocks_execution: true`
   - Returns: `"ERROR: QUERY_SPEC_INCOMPLETE"`

---

## Why This Happens

The error occurs when:

1. **File doesn't exist**: The requested file path doesn't exist in the data directory
   - Example: `data/sqlselect/sales.csv` doesn't exist
   - Investigation step (`list_dir`) fails to find it

2. **Path mismatch**: The Decider infers a path that doesn't match actual files
   - Example: Decider sets `start_table.path = "sqlselect/sales.csv"`
   - But actual file is at `ECommerce/sample_sales_data.csv`

3. **Investigation fails**: The investigation plan cannot resolve the blocking gap
   - `list_dir` doesn't find matching file
   - `inspect_table` fails because path is invalid
   - No fallback mechanism to ask user

---

## How to Fix

### Option 1: Ensure File Exists
- Verify the file exists at the expected path
- Check file naming matches what Decider expects
- Ensure proper directory structure

### Option 2: Improve Domain Configuration
- Add better `default_start_table_hint` in domain file
- Document actual file paths in domain configuration
- Provide clearer table name mappings

### Option 3: Better Error Handling
- Executor should detect investigation failure
- Should transition to `ASK_USER` when file not found
- Should provide clearer error message to user

---

## Related Code Locations

1. **Decider Prompt**: `external/config/prompts/decider.md`
   - Lines 115-122: Required minimum fields
   - Lines 86-91: Investigation plan triggers
   - Line 236: Blocking execution fields

2. **SQL Generator**: `external/tools/parquet_agent/nl_to_sql_planner.py`
   - Lines 100-103: Blocking gap check

3. **SQL Gate**: `external/agent/sql_gate.py`
   - `spec_ready_for_sql()`: Validates spec completeness

4. **Executor**: `external/agent/executor_nodes.py`
   - `investigation_node()`: Runs investigation plan
   - `sql_generation_node()`: Generates SQL after gate check

---

## Summary

The `QUERY_SPEC_INCOMPLETE` error is triggered by:

1. **Decider rule**: `start_table.path` is required (line 119)
2. **Status rule**: `start_table_grain` blocks execution (line 236)
3. **Investigation rule**: Missing blocking fields trigger investigation (lines 86-91)
4. **SQL generator rule**: Blocks SQL if gaps remain (nl_to_sql_planner.py:100-103)

The error occurs when investigation cannot resolve the missing `start_table.path`, leaving it in `"missing"` status with `blocks_execution: true`.

