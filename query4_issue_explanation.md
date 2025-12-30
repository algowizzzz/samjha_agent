# Query 4 Issue Explanation: `output_shape` Schema Validation Error

## The Problem

**Query**: "top 5 customers by total purchases"  
**Error**: `Validation error: Additional properties are not allowed ('limit', 'order_by', 'order_direction' were unexpected) (path: query_spec.output_shape)`

## Root Cause

The Decider LLM is trying to add `limit`, `order_by`, and `order_direction` fields to `output_shape`, but the schema only allows two fields:

**Allowed Schema** (`query_spec.schema.json`):
```json
"output_shape": {
  "type": "object",
  "additionalProperties": false,
  "required": ["type", "columns"],
  "properties": {
    "type": { "type": "string" },
    "columns": { "type": "array", "items": { "type": "string" } }
  }
}
```

**What Decider is trying to output** (INVALID):
```json
"output_shape": {
  "type": "table",
  "columns": ["customer", "total_purchases"],
  "limit": 5,                    // ❌ NOT ALLOWED
  "order_by": "total_purchases", // ❌ NOT ALLOWED
  "order_direction": "DESC"      // ❌ NOT ALLOWED
}
```

## Why This Happens

When the user asks for "top 5 customers", the Decider correctly recognizes:
- This is a ranking/limiting query
- It needs ORDER BY and LIMIT in SQL
- It tries to capture this intent in `output_shape`

However, `output_shape` is meant to describe the **structure** of the output (table/list/single value), not the **sorting/limiting behavior**.

## The Solution

**"Top N" and ordering information should NOT be in `output_shape`**. Instead:

1. **Capture in `business_question`**: The text "top 5 customers by total purchases" should remain in `business_question`
2. **SQL Planner handles it**: The SQL planner (`nl_to_sql_planner`) reads `business_question` and generates `ORDER BY` and `LIMIT` clauses based on the natural language

**Example of Query 1 (which succeeded)**:
- Query: "top products by revenue"
- `business_question`: "Show top products by revenue"
- `output_shape`: `null` (or minimal `{"type": "table", "columns": ["product", "revenue"]}`)
- SQL generated correctly includes: `ORDER BY revenue DESC LIMIT 10`

## What Needs to be Fixed

The Decider prompt needs explicit guidance that:

1. **`output_shape` must ONLY contain `type` and `columns`**
2. **Never add `limit`, `order_by`, `order_direction` to `output_shape`**
3. **"Top N" information should be captured in `business_question` text only**
4. **The SQL planner will infer ORDER BY/LIMIT from `business_question`**

## Current Status

- ✅ Query 1 ("top products by revenue") - Works because it doesn't populate invalid fields in `output_shape`
- ❌ Query 4 ("top 5 customers by total purchases") - Fails because Decider tries to add `limit`/`order_by` to `output_shape`

This is a **prompt engineering issue**, not a schema design issue. The schema is correct - the Decider just needs better guidance on what fields belong where.

