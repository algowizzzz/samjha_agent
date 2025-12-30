# Test Results Summary - ecommerce_advanced Agent

**Test Run Date**: 2025-12-29  
**Total Queries**: 10  
**Success Rate**: 5/10 (50%)

---

## Overall Results

| Status | Count | Percentage |
|--------|-------|------------|
| SUCCESS | 5 | 50% |
| ERROR | 4 | 40% |
| ASK_USER | 1 | 10% |

---

## Successful Queries (5)

### ✅ Query 1: "top products by revenue"
- **Status**: SUCCESS
- **SQL**: Correctly uses `SUM(quantity * price) AS revenue` ✓
- **Issue**: Uses single table (feb012024_sales_feb012024) instead of UNION ALL across all months
- **Aggregation Plan**: `single_table` (should probably be union_all_then_group for comprehensive results)

### ✅ Query 3: "inventory stock value month over month"
- **Status**: SUCCESS
- **SQL**: Correctly uses `SUM(stock_quantity * unit_cost) AS stock_value` ✓
- **UNION ALL**: Correctly implements UNION ALL across all months ✓
- **Aggregation Plan**: Correctly set to `union_all_then_group` ✓
- **Perfect example** of how multi-month queries should work!

### ✅ Query 5: "sales revenue by category for Electronics in North region"
- **Status**: SUCCESS
- **SQL**: Correctly uses `SUM(quantity * price) AS revenue` ✓
- **Filters**: Correctly applies category='Electronics' AND region='North' ✓
- **Issue**: Uses single table instead of UNION ALL (might be intentional for performance)

### ✅ Query 9: "sales revenue and order count by category"
- **Status**: SUCCESS
- **SQL**: Correctly implements multiple metrics:
  - `SUM(quantity * price) AS revenue` ✓
  - `COUNT(DISTINCT order_id) AS order_count` ✓
- **Multiple metrics**: Working correctly ✓

### ✅ Query 10: "total stock value by supplier"
- **Status**: SUCCESS
- **SQL**: Correctly uses `SUM(stock_quantity * unit_cost) AS stock_value` ✓
- **Supplier dimension**: Working correctly ✓

---

## Failed Queries (4)

### ❌ Query 2: "revenue by region and category for January"
- **Status**: ERROR
- **Root Cause**: Decider validation error - `fills_gap` is an array `['dimensions.region', 'dimensions.category']` instead of string
- **Error**: `Validation error: ['dimensions.region', 'dimensions.category'] is not of type 'string' (path: investigation_plan.1.fills_gap)`
- **Fix Needed**: Decider prompt needs to specify that `fills_gap` should be a single string, not an array

### ❌ Query 4: "top 5 customers by total purchases"
- **Status**: ERROR
- **Root Cause**: Decider validation error - `output_shape` contains unexpected properties (`limit`, `order_by`, `order_direction`)
- **Error**: `Validation error: Additional properties are not allowed ('limit', 'order_by', 'order_direction' were unexpected) (path: query_spec.output_shape)`
- **Fix Needed**: Schema validation or Decider prompt needs to handle `limit` differently (not in output_shape)

### ❌ Query 7: "products with low stock (below reorder level) in March"
- **Status**: ERROR
- **Root Cause**: Decider validation error - `fills_gap` is an array instead of string
- **Error**: `Validation error: ['dimensions.product', 'metrics.stock_quantity', 'metrics.reorder_level'] is not of type 'string'`
- **Fix Needed**: Decider should output single `fills_gap` string, not array

### ❌ Query 8: "revenue by region month over month"
- **Status**: ERROR
- **Root Cause**: Decider validation error - `fills_gap` is an array instead of string
- **Error**: `Validation error: ['dimensions.region', 'dimensions.report_date', 'start_table_grain'] is not of type 'string'`
- **Fix Needed**: Decider should output single `fills_gap` string per investigation step

---

## ASK_USER Query (1)

### ❓ Query 6: "average order value by customer tier"
- **Status**: ASK_USER
- **Query Spec**: Correctly identifies customer_tier dimension and avg_order_value metric
- **Issue**: Requires join between sales and customer tables, but Decider is asking for clarification
- **SQL Error** (from logs): `Referenced column "customer_tier" not found in FROM clause!`
- **Fix Needed**: Decider should recognize that customer_tier requires a join and either:
  - Plan the join automatically, OR
  - Ask user more clearly about the join requirement

---

## Key Findings

### ✅ Working Well

1. **Metric definition usage**: All successful queries correctly use `metric.definition` (e.g., `SUM(quantity * price)`) instead of `metric.name`
2. **UNION ALL for multi-month**: Query 3 demonstrates perfect UNION ALL implementation
3. **Multiple metrics**: Query 9 successfully handles multiple metrics in one query
4. **Filters**: Query 5 correctly applies multiple filters
5. **Aggregation plan preservation**: Query 3 shows aggregation_plan is being preserved correctly

### ❌ Issues Found

1. **Schema Validation Errors** (4 queries):
   - `fills_gap` field is being output as an array instead of a string
   - `output_shape` contains unexpected properties (`limit`, `order_by`)
   - **Fix**: Update Decider prompt to ensure `fills_gap` is always a single string

2. **Join Handling** (1 query):
   - Query 6 requires a join but Decider asks for clarification instead of planning the join
   - **Fix**: Decider should recognize join requirements from dimension location (customer_tier is in customer table)

3. **Single vs Multi-month Strategy** (Query 1):
   - "top products by revenue" uses single month instead of aggregating across all months
   - **Question**: Is this intentional (performance optimization) or should it aggregate across all months?

---

## Recommendations

### High Priority Fixes

1. **Fix `fills_gap` validation error**: Update Decider prompt to explicitly state that `fills_gap` must be a single string per investigation step, not an array
2. **Fix `output_shape` schema**: Either update schema to allow `limit`/`order_by` or update Decider to put these in a different field
3. **Improve join detection**: Decider should recognize when a dimension requires a join (e.g., customer_tier requires sales↔customer join)

### Medium Priority Improvements

1. **Multi-month strategy**: Clarify when to use UNION ALL vs single table (performance vs completeness tradeoff)
2. **Aggregation plan for single-table queries**: Query 1 and 5 should probably have aggregation_plan set (even if single_table type)

---

## Test Files Generated

- `test_ecommerce_queries_log.json`: Full detailed results with all query specs, SQL, errors
- `test_run_output.log`: Console output from test run
- `test_queries_ecommerce_advanced.md`: Original test query list with expected behaviors

