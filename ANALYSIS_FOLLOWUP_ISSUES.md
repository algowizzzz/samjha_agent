# Analysis: Follow-up Query Issues

## Problem Summary

### Query 1: "top primary limits"
**Generated SQL:** `SELECT * FROM limits_data LIMIT 10`

**Issues:**
1. ❌ **Missing filter**: Should filter `WHERE limit_class = 'Primary'` but didn't
2. ❌ **Missing sort**: "top" implies ordering by utilization/exposure DESC, but no ORDER BY
3. ❌ **Default not applied**: Config says "PRIMARY limits if not specified" but not enforced

**Expected SQL:**
```sql
SELECT * FROM limits_data 
WHERE limit_class = 'Primary' 
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY utilization DESC, exposure_amt DESC 
LIMIT 10
```

---

### Query 2: "what is the sum of exposure across of these limits"
**Generated SQL:** `SELECT * FROM limits_data LIMIT 10` (SAME as Query 1!)

**Issues:**
1. ❌ **Aggregation ignored**: Requested SUM but generated SELECT *
2. ❌ **Context lost**: "these limits" refers to previous query results, but ignored
3. ❌ **No conversation history usage**: Should use previous query's filters/context
4. ❌ **Wrong query type**: Should be aggregation, not row retrieval

**Expected SQL:**
```sql
SELECT SUM(exposure_amt) as total_exposure
FROM limits_data 
WHERE limit_class = 'Primary' 
  AND date = (SELECT MAX(date) FROM limits_data)
```

---

## Root Causes

### 1. SQL Generator Not Applying Defaults
**Location:** `config/agent/queryagent_planner.json` - `sql_generator_system` prompt

**Problem:** The prompt says to apply defaults, but the LLM is not consistently doing so:
- "Level: Filter limit_level = 'Primary' if not specified" → Not applied
- "Sorting: ORDER BY utilization DESC when querying limits" → Not applied
- "Date: Use WHERE date = (SELECT MAX(date) FROM table)" → Not applied

**Fix Needed:** Make defaults more explicit in the prompt, or enforce them in code.

---

### 2. Conversation History Not Effectively Used
**Location:** `tools/impl/nl_to_sql_planner.py` - `_llm_plan()` method

**Problem:** 
- Conversation history is passed to the LLM but not being used to understand:
  - Pronouns ("these", "those", "them") referring to previous results
  - Follow-up aggregations ("sum of these", "count of those")
  - Context from previous queries

**Current Format:**
```
Previous Conversation:
Turn 1:
User Query: top primary limits
SQL Executed: SELECT * FROM limits_data LIMIT 10
Result Table (10 rows):
...
```

**Issue:** The LLM sees the history but doesn't connect "these limits" to the previous query's results.

**Fix Needed:** 
- Enhance prompt to explicitly instruct LLM to use conversation history
- Add examples of follow-up queries in the prompt
- Make the connection between "these/those" and previous results more explicit

---

### 3. Aggregation Requests Not Recognized
**Location:** `sql_generator_system` prompt

**Problem:** The prompt doesn't explicitly handle aggregation requests:
- "sum of exposure" → Should generate `SELECT SUM(exposure_amt)`
- "count of these" → Should generate `SELECT COUNT(*)` with previous filters
- "average utilization" → Should generate `SELECT AVG(utilization)`

**Fix Needed:** Add explicit instructions for handling aggregation keywords (SUM, COUNT, AVG, etc.)

---

### 4. Follow-up Query Context Not Preserved
**Location:** `agent/graph_nodes.py` - `generate_sql_node()`

**Problem:** When processing follow-up queries:
- The previous query's filters are not automatically included
- The conversation history shows the data but not the SQL filters
- The LLM has to infer context from the table data, not the SQL WHERE clause

**Fix Needed:** 
- Include previous SQL WHERE clauses in conversation history
- Make it explicit that follow-up queries should inherit previous filters
- Add examples: "If user says 'sum of these', use the WHERE clause from previous query"

---

## Recommended Fixes

### Fix 1: Enhance SQL Generator Prompt
Add explicit aggregation handling:
```
If user requests aggregation (sum, count, average, etc.):
- Use aggregation function: SUM(column), COUNT(*), AVG(column)
- Apply WHERE filters from conversation history if user says "these/those"
- Return single row with aggregated value
```

### Fix 2: Improve Conversation History Format
Include SQL filters explicitly:
```
Previous Query: "top primary limits"
SQL Executed: SELECT * FROM limits_data WHERE limit_class = 'Primary' ... LIMIT 10
Filters Applied: limit_class = 'Primary', date = MAX(date)
Result: 10 rows
```

### Fix 3: Add Follow-up Query Examples
In `sql_generator_system` prompt, add:
```
FOLLOW-UP QUERIES:
If user says "sum of these limits" or "count of those":
1. Extract WHERE clause from previous query in conversation history
2. Apply same filters to new aggregation query
3. Example: Previous had "WHERE limit_class = 'Primary'", follow-up should include same filter
```

### Fix 4: Enforce Defaults in Code
If LLM doesn't apply defaults, enforce them in `generate_sql_node()`:
- Check if query mentions "primary" → add `WHERE limit_class = 'Primary'`
- Check if query mentions "top" → add `ORDER BY utilization DESC`
- Check if no date specified → add `WHERE date = (SELECT MAX(date) FROM table)`

---

## Priority

1. **HIGH**: Fix aggregation recognition (Query 2 completely wrong)
2. **HIGH**: Fix conversation history usage (follow-ups not working)
3. **MEDIUM**: Enforce defaults for "primary" and "top" filters
4. **LOW**: Improve prompt examples

