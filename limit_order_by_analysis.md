# Analysis: Should `limit`, `order_by`, `order_direction` be in the Schema?

## Your Concern (Valid!)
You're worried that if "top 5" information isn't explicitly captured, the SQL might return `LIMIT 100` or more rows instead of the requested 5.

## Current State Analysis

### 1. What Actually Happens Now

**Query 1** ("top products by revenue") - **SUCCEEDED**:
- SQL generated: `ORDER BY revenue DESC` (no LIMIT clause)
- Business Question: "Show top products by revenue"
- **Problem**: SQL has no LIMIT, so it could return all products!

**Query Safety Validator**:
- Checks if SQL has `LIMIT` clause
- If missing, flags: `NO_LIMIT:will_enforce_{max_rows}`
- Default `max_rows` = **1000** (from `policy_limits`)

**execute_sql Tool**:
- Takes `max_rows` parameter (default: 100 if not provided)
- But the actual SQL execution might return all rows if SQL has no LIMIT
- Need to check if execute_sql enforces LIMIT when SQL doesn't have it

### 2. The Problem

**Current Flow**:
1. User asks: "top 5 customers by total purchases"
2. Decider tries to add `limit: 5` to `output_shape` → **VALIDATION ERROR** (schema doesn't allow it)
3. Decider falls back to just putting "top 5" in `business_question` text
4. SQL planner reads `business_question` → **might miss the "5"** and generate no LIMIT or wrong LIMIT
5. Result: Returns 1000 rows (default max) instead of 5 ❌

**Evidence**:
- Query 1 succeeded but SQL has no LIMIT clause (just ORDER BY)
- This confirms the SQL planner doesn't reliably extract limit from natural language

## Should They Be in Schema?

### **YES - They Should Be in Schema**

**Reasoning**:

1. **Semantic Separation**:
   - `output_shape` = **STRUCTURE** (what columns, table/list/scalar)
   - `limit`/`order_by` = **BEHAVIOR** (how results are presented/ordered)

2. **Explicit is Better Than Implicit**:
   - SQL planner shouldn't guess "top 5" means LIMIT 5
   - Explicit fields prevent ambiguity

3. **Current System Needs It**:
   - Query 1 proves SQL planner doesn't always add LIMIT
   - Without explicit limit, defaults to 1000 rows (bad UX)

### Where Should They Go?

**Option A: Add to `query_spec` as separate fields** (RECOMMENDED)
```json
{
  "query_spec": {
    "business_question": "...",
    "output_shape": { "type": "table", "columns": [...] },
    "sorting": {
      "order_by": ["revenue"],  // array to support multiple columns
      "direction": "DESC"
    },
    "limit": 5  // optional integer
  }
}
```

**Option B: Add to `output_shape`** (NOT RECOMMENDED)
- Mixes structure with behavior
- Breaks semantic clarity
- But technically possible if schema allows `additionalProperties: true`

**Option C: Keep only in `business_question`** (CURRENT - PROBLEMATIC)
- SQL planner must parse natural language
- Query 1 shows this doesn't work reliably
- No guarantee "top 5" becomes LIMIT 5

## Recommendation

**Add to `query_spec.schema.json` as separate optional fields**:

```json
{
  "properties": {
    // ... existing fields ...
    "sorting": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "order_by": {
          "type": "array",
          "items": { "type": "string" }
        },
        "direction": {
          "type": "string",
          "enum": ["ASC", "DESC"]
        }
      }
    },
    "limit": {
      "type": ["integer", "null"],
      "minimum": 1
    }
  }
}
```

**Benefits**:
1. ✅ Explicit contract - no guessing
2. ✅ SQL planner gets exact instructions
3. ✅ Prevents returning 1000 rows when user wants 5
4. ✅ Separate from `output_shape` (clean semantics)
5. ✅ Optional fields (backward compatible)

**Trade-offs**:
- Slightly more verbose schema
- Decider must extract limit/order from user query (but it's already doing this, just putting it in wrong place)

## Conclusion

**YES, `limit` and `order_by`/`order_direction` SHOULD be in the schema**, but **NOT in `output_shape`**. They should be separate optional fields in `query_spec`:

- `sorting`: { "order_by": [...], "direction": "ASC|DESC" }
- `limit`: integer (optional, nullable)

This ensures:
1. SQL planner gets explicit instructions (no parsing ambiguity)
2. User's "top 5" request actually returns 5 rows (not 1000)
3. Clean semantic separation (structure vs behavior)
4. Backward compatible (optional fields)

