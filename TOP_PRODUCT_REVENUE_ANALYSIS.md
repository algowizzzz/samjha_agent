# Analysis: Why "Top Product by Revenue" Now Asks for Clarifications

## Executive Summary

The query "top products by revenue" worked successfully on **2025-12-30** (test log shows SUCCESS), but now asks for clarifications. The root cause is **stricter enforcement of `start_table_grain` validation** combined with the LLM not consistently following grain inference rules in the Decider prompt.

## Evidence Base

### 1. Historical Test Results (When It Worked)

**Test Run: 2025-12-30T10:50:54**
- **Query**: "top products by revenue"
- **Status**: SUCCESS ✅
- **SQL Generated**: `SELECT product, SUM(quantity * price) AS revenue FROM sales_feb012024 WHERE report_date >= CURRENT_DATE - INTERVAL '30' DAY GROUP BY product ORDER BY revenue DESC LIMIT 10;`
- **Query Spec**:
  - `business_question`: "Show top products by revenue"
  - `start_table`: `{name: "feb012024_sales_feb012024", path: "feb012024/sales_feb012024.csv"}`
  - `dimensions`: `["product"]`
  - `metrics`: `[{name: "revenue", definition: "Sum of (quantity * price)"}]`
  - `time`: `{column: "report_date", rule: "last_n_days", n_days: 30}`

**Key Observation**: The query spec had all required fields populated, including grain (inferred as "one row per product" from dimensions).

### 2. Domain Configuration (Still Supports It)

**File**: `external/config/domains/ecommerce_advanced_domain.md`

**Section 11.1** (User Query → Query Spec Examples):
```
| "top products by sales" | *_sales_* | [{"name": "revenue", "definition": "Sum of (quantity * price)"}] | ["product"] | - | [] | {"order_by": ["revenue"], "direction": "DESC"} | 10 |
```

**Section 11.0.1** (Table Selection Logic):
```
| "sales", "revenue", "orders", "products", "top products" | *_sales_* | Sales/revenue metrics |
```

**Section 11.0** (Default behaviors):
```
- "top products" / "best products" → revenue metric, DESC sort
- No limit specified for "top N" → default to 10
```

**Conclusion**: Domain configuration explicitly supports "top products by revenue" and provides clear defaults.

### 3. SQL Gate Validation (The Enforcement Point)

**File**: `external/agent/sql_gate.py`

**Function**: `spec_ready_for_sql()`

**Critical Check** (lines 84-89):
```python
# Check start_table_grain
start_table_grain_status = query_spec_status.get("start_table_grain", {})
if start_table_grain_status.get("status") in ["missing", "conflict"]:
    return False
if start_table_grain_status.get("blocks_execution", False):
    return False
```

**Impact**: If `start_table_grain` status is `"missing"` OR `blocks_execution: true`, SQL generation is **blocked**.

### 4. Decider Prompt Instructions (The Inference Rules)

**File**: `external/config/prompts/decider.md`

**Section: "CRITICAL: When inferring grain"** (lines 391-427):

**Priority Order for Grain Inference**:
1. **First**: Check query text patterns ("by X", "top N", etc.)
   - Example: "top N products" → grain: "one row per product"
2. **Second**: Check domain.md grain examples (Section 4: Core entities)
3. **Third**: Derive from dimensions array
   - If dimensions = ["product"] → grain: "one row per product"
4. **Last**: If still unclear, mark as `missing` and add `inspect_table` step

**Status Rules** (line 423-427):
- If inferred from query text → status: `inferred`, source: `user`
- If from domain.md → status: `inferred`, source: `domain_md`
- If from dimensions → status: `inferred`, source: `user`
- **If missing** → status: `missing`, source: `rule`, **blocks_execution: `true`**

**Key Instruction** (line 414):
> "Last: If still unclear, mark as `missing` and add `inspect_table` step to fills_gap: 'start_table_grain'"

### 5. Required Minimum for Execution

**File**: `external/config/prompts/decider.md` (lines 852-861)

**REQUIRED MINIMUM FOR EXECUTION**:
- `business_question`
- `start_table.path`
- **`grain`** ← Critical field
- `metrics`
- `time.column` **OR** `time.rule = "no_time"`

**Verification**: "Verification happens in the Executor" (but SQL gate blocks before SQL generation if `start_table_grain` is missing).

## Root Cause Analysis

### The Problem Chain

1. **Query**: "top products by revenue"
2. **Expected Behavior** (per prompt instructions):
   - Pattern "top products" → grain: "one row per product" (line 394)
   - OR dimensions = ["product"] → grain: "one row per product" (line 407)
   - Status should be: `inferred`, source: `user` or `domain_md`
   - `blocks_execution`: `false`

3. **Actual Behavior** (current):
   - LLM marks `start_table_grain` as `missing` with `blocks_execution: true`
   - OR marks status as `inferred` but sets `blocks_execution: true` (incorrect)
   - SQL gate blocks execution
   - Decider outputs `ASK_USER` instead of `EXECUTE`

### Why This Happens Now

**Hypothesis 1: LLM Model Drift/Conservatism**
- The LLM is being more conservative and not following the inference priority order
- It's skipping the inference steps and going directly to "mark as missing"
- This could be due to:
  - Model updates/changes
  - Temperature/sampling changes
  - Prompt length/complexity causing the LLM to miss the inference rules

**Hypothesis 2: Prompt Ambiguity**
- The prompt says "If still unclear, mark as missing" (line 414)
- The LLM might interpret "top products by revenue" as "unclear" even though the prompt provides clear patterns
- The prompt has multiple inference methods, and the LLM might be confused about which to use

**Hypothesis 3: SQL Gate Enforcement Awareness**
- The SQL gate strictly enforces `start_table_grain` validation
- The LLM might be "aware" (through training or context) that `start_table_grain` is critical
- This makes it more cautious and likely to ask for clarification rather than infer

**Hypothesis 4: Missing Explicit Example**
- The prompt has examples for "top products by sales" but not "top products by revenue"
- The LLM might not recognize "revenue" as equivalent to "sales" in this context
- Domain_md Section 11.1 shows "top products by sales" but the user query is "top products by revenue"

### Most Likely Root Cause

**Combination of Hypothesis 2 + Hypothesis 4**:

1. The prompt provides inference rules, but the LLM is not consistently applying them
2. The domain_md example shows "top products by sales" but the query is "top products by revenue"
3. The LLM doesn't recognize that "revenue" = "sales" in this context (even though domain_md Section 11.0 says "sales" → revenue metric)
4. The LLM defaults to "mark as missing" when it can't find an exact match

## Concrete Evidence of the Issue

### Evidence 1: Test Log Shows It Worked Before
- **File**: `test_ecommerce_queries_log.json`
- **Date**: 2025-12-30
- **Status**: SUCCESS
- **Proof**: The query executed successfully with proper grain inference

### Evidence 2: Domain Configuration Supports It
- **File**: `external/config/domains/ecommerce_advanced_domain.md`
- **Section 11.1**: Explicit example for "top products by sales"
- **Section 11.0**: Default behavior: "top products" → revenue metric, DESC sort
- **Proof**: Domain configuration has all necessary information

### Evidence 3: SQL Gate Blocks Missing Grain
- **File**: `external/agent/sql_gate.py`
- **Lines 84-89**: Strict validation that blocks if `start_table_grain` is missing or `blocks_execution: true`
- **Proof**: Code-level enforcement that prevents execution

### Evidence 4: Prompt Has Inference Rules But LLM Doesn't Follow Them
- **File**: `external/config/prompts/decider.md`
- **Lines 391-427**: Clear inference rules for grain
- **Line 394**: "top N products" → grain: "one row per product"
- **Proof**: Prompt instructions exist but aren't being followed consistently

## What Changed (No Code Changes, Just Analysis)

### No Code Changes Detected
- SQL gate validation logic: **Unchanged** (still enforces `start_table_grain`)
- Domain configuration: **Unchanged** (still has examples)
- Schema validation: **Unchanged** (still requires grain)

### What Likely Changed
1. **LLM Behavior**: The LLM is now more conservative and not following grain inference rules
2. **Prompt Interpretation**: The LLM might be interpreting "top products by revenue" differently than "top products by sales"
3. **Model Updates**: If the underlying LLM model was updated, it might have different inference behavior

## Recommendations

### Immediate Fix (Prompt Engineering)

**Add explicit example to Decider prompt**:

In the "CRITICAL: When inferring grain" section, add:
```
**Example for "top products by revenue":**
- Query: "top products by revenue"
- Pattern match: "top products" → grain: "one row per product"
- Dimensions: ["product"] → confirms grain: "one row per product"
- Status: `inferred`, source: `user`, blocks_execution: `false`
- Action: `EXECUTE` (do NOT ask for clarification)
```

**Strengthen the inference priority**:

Change line 414 from:
> "Last: If still unclear, mark as `missing`..."

To:
> "Last: If still unclear after checking ALL three methods above, mark as `missing`... **For 'top products by revenue', this should NEVER be unclear - use pattern matching or dimensions array.**"

### Long-term Fix (Domain Configuration)

**Add explicit example to domain_md Section 11.1**:
```
| "top products by revenue" | *_sales_* | [{"name": "revenue", "definition": "Sum of (quantity * price)"}] | ["product"] | - | [] | {"order_by": ["revenue"], "direction": "DESC"} | 10 |
```

This provides an exact match for the user query.

## Conclusion

**Root Cause**: The LLM is not consistently following the grain inference rules in the Decider prompt. It's marking `start_table_grain` as `missing` or setting `blocks_execution: true` instead of inferring grain from the query pattern "top products" → "one row per product".

**Evidence**:
1. ✅ Test log shows it worked on 2025-12-30
2. ✅ Domain configuration supports it
3. ✅ SQL gate enforces validation (blocks if grain missing)
4. ✅ Prompt has inference rules but LLM doesn't follow them consistently

**Solution**: Strengthen the prompt with explicit examples and make the inference priority more explicit. Add "top products by revenue" to domain_md examples for exact matching.

