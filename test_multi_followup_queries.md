# Multi-Turn Follow-Up Test Sequence

## Test Scenario
5 queries where Query 5 references Query 2 (not the immediate prior Query 4).
This tests:
- conversation_history can handle 5 turns
- prior_query_spec is from latest (Query 4)
- Query 5 can reference earlier queries from conversation_history

---

## Query Sequence

### Query 1: Initial Query (NEW_QUERY)
**Query:** "Show me total revenue by region for orders in January 2024"

**Expected:**
- query_type: NEW_QUERY
- dimensions: ["region"]
- metrics: [{"name": "revenue", "definition": "Sum of (quantity * price)"}]
- time: date_range (2024-01-01 to 2024-01-31)
- filters: []

**Expected Results (from data):**
- East: $926.50
- North: $3,875.00
- South: $5,301.00
- West: $1,333.00
- **Total rows: 4**

---

### Query 2: First Follow-up (FOLLOW_UP)
**Query:** "What about breakdown by product category too?"

**Expected:**
- query_type: FOLLOW_UP
- signals: ["what about" = continuation, "too" = addition]
- dimensions: ["region", "category"] (added category to prior region)
- metrics: [revenue] (preserved)
- time: January 2024 (preserved)
- filters: []

**Expected Results (from data):**
- East | Electronics: $926.50
- North | Electronics: $2,625.00
- North | Furniture: $1,250.00
- South | Electronics: $4,901.00
- South | Furniture: $400.00
- West | Electronics: $408.00
- West | Furniture: $925.00
- **Total rows: 7**

---

### Query 3: Second Follow-up (FOLLOW_UP)
**Query:** "Now filter to only Electronics category"

**Expected:**
- query_type: FOLLOW_UP
- signals: ["now" = continuation, "only" = filter signal]
- dimensions: ["region"] (category removed since filtering to single value)
- metrics: [revenue] (preserved)
- time: January 2024 (preserved)
- filters: [{"field": "category", "operator": "=", "value": "Electronics"}]

**Expected Results (from data):**
- East: $926.50
- North: $2,625.00
- South: $4,901.00
- West: $408.00
- **Total rows: 4** (only Electronics, all regions)

---

### Query 4: Third Follow-up (FOLLOW_UP)
**Query:** "Show me the top 3 products by sales quantity"

**Expected:**
- query_type: FOLLOW_UP
- signals: ["show me" = continuation, "top 3" = ranking]
- dimensions: [] (no grouping, just top products)
- metrics: [{"name": "quantity", "definition": "SUM(quantity)"}] (changed from revenue)
- time: January 2024 (preserved from prior)
- filters: [] (removed category filter, now showing all)
- aggregation_plan: ORDER BY SUM(quantity) DESC LIMIT 3

**Expected Results (from data):**
- 1. Mouse: 6 units
- 2. Monitor: 6 units
- 3. Chair: 6 units
- **Total rows: 3**

---

### Query 5: Fourth Follow-up - References Query 2 (FOLLOW_UP)
**Query:** "What was the revenue for those categories we looked at earlier?"

**Expected:**
- query_type: FOLLOW_UP
- signals: ["what was" = continuation, "those categories" = pronoun reference to Query 2, "earlier" = time reference]
- Should understand "those categories" refers to Query 2's category breakdown
- dimensions: ["category"] (from Query 2, not Query 4)
- metrics: [revenue] (back to revenue from Query 4's quantity)
- time: January 2024 (preserved)
- filters: [] (no category filter, showing all categories)

**Expected Results (from data):**
- Electronics: $8,860.50
- Furniture: $2,575.00
- **Total rows: 2** (referencing Query 2's category dimension)

**Key Test:** Query 5's prior_query_spec will be from Query 4 (top 3 products), but it should use conversation_history to understand "those categories" refers to Query 2.

---

## Test Execution Order

1. Execute Query 1 → Get result, save to conversation_history[0]
2. Execute Query 2 with prior_state from Query 1 → Get result, save to conversation_history[1]
3. Execute Query 3 with prior_state from Query 2 → Get result, save to conversation_history[2]
4. Execute Query 4 with prior_state from Query 3 → Get result, save to conversation_history[3]
5. Execute Query 5 with prior_state from Query 4 → Should reference Query 2 from conversation_history

---

## Success Criteria

✅ Query 1: NEW_QUERY, revenue by region
✅ Query 2: FOLLOW_UP, adds category dimension
✅ Query 3: FOLLOW_UP, filters to Electronics
✅ Query 4: FOLLOW_UP, changes to top 3 products by quantity
✅ Query 5: FOLLOW_UP, correctly references Query 2's categories despite prior_query_spec being from Query 4

---

## Expected conversation_history at Query 5

```json
[
  {
    "query": "Show me total revenue by region for orders in January 2024",
    "sql": "SELECT region, SUM(quantity * price) AS revenue FROM...",
    "response": "...",
    "status": "SUCCESS"
  },
  {
    "query": "What about breakdown by product category too?",
    "sql": "SELECT region, category, SUM(quantity * price) AS revenue FROM...",
    "response": "...",
    "status": "SUCCESS"
  },
  {
    "query": "Now filter to only Electronics category",
    "sql": "SELECT region, SUM(quantity * price) AS revenue FROM... WHERE category = 'Electronics'",
    "response": "...",
    "status": "SUCCESS"
  },
  {
    "query": "Show me the top 3 products by sales quantity",
    "sql": "SELECT product, SUM(quantity) AS quantity FROM... ORDER BY SUM(quantity) DESC LIMIT 3",
    "response": "...",
    "status": "SUCCESS"
  }
]
```

## Expected prior_query_spec at Query 5

```json
{
  "business_question": "Show top 3 products by sales quantity",
  "dimensions": [],
  "metrics": [{"name": "quantity", "definition": "SUM(quantity)"}],
  "time": {"column": "order_date", "rule": "date_range", "start": "2024-01-01", "end": "2024-01-31"},
  "filters": []
}
```

**Note:** This is from Query 4, but Query 5 should use conversation_history to understand "those categories" refers to Query 2.

