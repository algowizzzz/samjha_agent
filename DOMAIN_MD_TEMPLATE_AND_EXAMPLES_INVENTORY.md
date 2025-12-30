# Domain MD Template and Examples Inventory

This document serves as a reference for:
1. All examples currently used in system prompts (for migration to domain_md)
2. A template for creating new domain_md files
3. A checklist to ensure nothing is missed when adding new agents

---

## COMPREHENSIVE INVENTORY: Examples in System Prompts

### **1. TABLE/VIEW NAME EXAMPLES**

| Prompt | Example | Type |
|--------|---------|------|
| decider.md | `jan012024_sales_jan012024` | Specific view |
| decider.md | `feb012024_sales_feb012024` | Specific view |
| decider.md | `feb012024_customer_feb012024` | Specific view |
| decider.md | `*_sales_*`, `*_customer_*` | Pattern |
| decider.md | `jan012024/sales_jan012024.csv` | Path |
| nl_to_sql_planner.md | `ECommerce/sample_sales_data.csv` | Path with folder |
| nl_to_sql_planner.md | `ECommerce/sample_customer_data.csv` | Path with folder |
| nl_to_sql_planner.md | `feb012024_sales_feb012024`, `feb012024_customer_feb012024` | Join example |
| nl_to_sql_planner.md | `jan012024_sales_jan012024 UNION ALL feb012024_sales... UNION ALL mar012024_sales...` | UNION example |

---

### **2. USER QUERY EXAMPLES**

| Prompt | Query Example | Purpose |
|--------|---------------|---------|
| decider.md | "Show me total revenue by region" | NEW_QUERY example |
| decider.md | "What about products too?" | FOLLOW_UP (add dimension) |
| decider.md | "now what was the order count for the ones from east region only" | FOLLOW_UP (filter + metric change) |
| decider.md | "what about last month instead" | FOLLOW_UP (time change) |
| decider.md | "revenue" | USER_ANSWER example |
| decider.md | "Show me customer signups by month for 2024" | NEW_QUERY (different entity) |
| decider.md | "top products by sales" | Entity inference example |
| decider.md | "average order value by customer_tier" | Join detection example |
| decider.md | "revenue by region" | Dimension from same table |
| ask_user.md | "what data do you have?" | Informational |
| ask_user.md | "what can you query for me?" | Informational |
| ask_user.md | "show top sales", "find customers by region" | Data analysis |
| ask_user.md | "hello", "how are you", "sky is blue" | Off-topic |

---

### **3. DIMENSION/COLUMN EXAMPLES**

| Prompt | Column | Used In |
|--------|--------|---------|
| decider.md | `region` | Dimension, filter, GROUP BY |
| decider.md | `category` | Dimension, filter |
| decider.md | `product` | Dimension |
| decider.md | `customer_tier` | Join dimension |
| decider.md | `report_date` | Time column, UNION GROUP BY |
| decider.md | `order_date` | Time column |
| decider.md | `customer_id` | Join key |
| decider.md | `month` | Dimension (derived) |
| nl_to_sql_planner.md | `customer_tier` | Join column |
| nl_to_sql_planner.md | `quantity`, `price` | Metric calculation |

---

### **4. METRIC DEFINITION EXAMPLES**

| Prompt | Metric | Definition |
|--------|--------|------------|
| decider.md | `revenue` | "Sum of (quantity * price)" |
| decider.md | `order_count` | "COUNT(DISTINCT order_id)" |
| decider.md | `signup_count` | "COUNT(DISTINCT customer_id)" |
| nl_to_sql_planner.md | `revenue` | "Sum of (quantity * price)" |
| nl_to_sql_planner.md | `avg_order_value` | "revenue / order_count" |

---

### **5. FILTER EXAMPLES**

| Prompt | Filter |
|--------|--------|
| decider.md | `{"field": "region", "op": "=", "value": "East"}` |
| decider.md | `{"field": "category", "operator": "=", "value": "Electronics"}` |
| decider.md | `WHERE category='Electronics'` |
| decider.md | `WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31'` |

---

### **6. DATE/TIME EXAMPLES**

| Prompt | Date/Value |
|--------|------------|
| decider.md | `"2024-01-01"`, `"2024-01-31"`, `"2024-12-31"` |
| decider.md | `30` (n_days) |
| decider.md | `"last_n_days"`, `"date_range"`, `"no_time"` |

---

### **7. SQL GENERATION EXAMPLES**

| Prompt | SQL Pattern |
|--------|-------------|
| nl_to_sql.md | `SUM(quantity * price) AS revenue` |
| nl_to_sql.md | `SUM(quantity * price) / COUNT(DISTINCT order_id) AS avg_order_value` |
| nl_to_sql.md | `FROM sample_sales_data` (view name) |
| nl_to_sql.md | `LEFT JOIN feb012024_customer_feb012024 c ON s.customer_id = c.customer_id` |
| nl_to_sql.md | `SELECT report_date, SUM(quantity * price) AS revenue FROM (...UNION ALL...) GROUP BY report_date` |
| nl_to_sql.md | `ORDER BY revenue DESC`, `LIMIT 5` |

---

### **8. JOIN EXAMPLES**

| Prompt | Join Pattern |
|--------|--------------|
| decider.md | `left_table: *_sales_*, right_table: *_customer_*` |
| decider.md | `on: {sales_view}.customer_id = {customer_view}.customer_id` |
| nl_to_sql.md | `FROM feb012024_sales_feb012024 s LEFT JOIN feb012024_customer_feb012024 c` |
| nl_to_sql.md | Table aliases: `s`, `c` |

---

### **9. GRAIN EXAMPLES**

| Prompt | Grain Pattern |
|--------|---------------|
| decider.md | "one row per region" |
| decider.md | "one row per product" |
| decider.md | "one row per customer" |
| decider.md | "one row per region-category combination" |
| decider.md | "one row per order" |
| decider.md | "one row per report_date" |

---

### **10. AGGREGATION PLAN EXAMPLES**

| Prompt | Pattern |
|--------|---------|
| decider.md | `aggregation_type: "union_all_then_group"` |
| decider.md | `union_strategy.pattern: "*_sales_*"` |
| decider.md | `group_by: ["report_date"]` |
| nl_to_sql.md | `SELECT * FROM view1 UNION ALL SELECT * FROM view2 UNION ALL SELECT * FROM view3` |

---

### **11. SORTING/LIMIT EXAMPLES**

| Prompt | Pattern |
|--------|---------|
| decider.md | `"top 5"`, `"first 10"` |
| decider.md | `sorting: {"order_by": ["revenue"], "direction": "DESC"}` |
| decider.md | `limit: 5` |
| nl_to_sql.md | `ORDER BY revenue DESC`, `LIMIT 5` |

---

### **12. ERROR MESSAGE EXAMPLES**

| Prompt | Message |
|--------|---------|
| decider.md | "I can't find `promo_code` in this dataset. Did you mean `product` or `category`?" |

---

### **13. DOMAIN KEY EXAMPLES**

| Prompt | Key |
|--------|-----|
| decider.md | `"ecomm"`, `"ecommerce_advanced"` |

---

## DOMAIN_MD TEMPLATE

### Section 11: Example Patterns (LLM Reference Examples)

This section should be added to every domain_md file to provide agent-specific examples that the LLM can reference. This replaces hardcoded examples in system prompts.

```markdown
## 11) Example Patterns (LLM Reference Examples)

This section provides concrete examples for the LLM to understand how to process queries for this specific dataset.

### 11.1 User Query → Query Spec Examples

These examples show how to translate user queries into query_spec fields.

| User Query | metrics | dimensions | filters | sorting | limit | grain |
|------------|---------|------------|---------|---------|-------|-------|
| "<example_query_1>" | [<metric_name>] | [<dim1>] | - | {"order_by": ["<metric>"], "direction": "DESC"} | <N> | "one row per <dim1>" |
| "<example_query_2>" | [<metric_name>] | [<dim1>, <dim2>] | - | - | - | "one row per <dim1>-<dim2> combination" |
| "<follow_up_query>" | [<metric_name>] | [<dim1>] | [{"field": "<col>", "op": "=", "value": "<val>"}] | - | - | "one row per <dim1>" |

### 11.2 Dimension Inference Examples

How to map natural language to dimension columns.

| User Language | Dimension Found | Column Used | Table |
|---------------|-----------------|-------------|-------|
| "by <natural_term>" | <dimension_name> | <column_name> | <table_pattern> |
| "top <entity_plural>" | <dimension_name> | <column_name> | <table_pattern> |
| "grouped by <term>" | <dimension_name> | <column_name> | <table_pattern> |

### 11.3 Metric Calculation Examples

How to translate metric definitions to SQL expressions.

| Metric Name | Definition (Natural) | SQL Expression |
|-------------|---------------------|----------------|
| <metric_1> | "<natural_language_definition>" | `<SQL_EXPRESSION>` |
| <metric_2> | "<formula_referencing_other_metrics>" | `<EXPANDED_SQL_EXPRESSION>` |

**Metric Expansion Example:**
- If `avg_order_value = revenue / order_count`
- And `revenue = Sum of (quantity * price)`
- And `order_count = Count of distinct order_id`
- Then SQL: `SUM(quantity * price) / COUNT(DISTINCT order_id) AS avg_order_value`

### 11.4 Filter Pattern Examples

How to translate filter language to filter objects.

| User Language | Filter Generated |
|---------------|------------------|
| "only <value>" | `{"field": "<column>", "op": "=", "value": "<value>"}` |
| "for <entity> <value>" | `{"field": "<column>", "op": "=", "value": "<value>"}` |
| "in <date_range>" | `{"field": "<date_col>", "op": "BETWEEN", "start": "<date1>", "end": "<date2>"}` |

### 11.5 Join Detection Examples

When queries require joining tables.

| Query Pattern | Start Table | Dimension Table | Join Required | Join From Section 8 |
|---------------|-------------|-----------------|---------------|---------------------|
| "<query_requiring_join>" | <start_pattern> | <dim_pattern> | Yes | `<left> JOIN <right> ON <condition>` |
| "<query_same_table>" | <pattern> | <same_pattern> | No | - |

### 11.6 UNION ALL Aggregation Examples

For datasets with multiple views that need to be combined.

| Query Type | View Pattern | Aggregation Plan | SQL Template |
|------------|--------------|------------------|--------------|
| "trend over time" | <pattern> | union_all_then_group | `SELECT <date_col>, ... FROM (<view1> UNION ALL <view2> UNION ALL ...) GROUP BY <date_col>` |
| "single period" | <specific_view> | single_table | `SELECT ... FROM <specific_view>` |

### 11.7 Sorting and Limit Examples

How to extract sorting/limit from user queries.

| User Language | sorting | limit |
|---------------|---------|-------|
| "top N by X" | `{"order_by": ["X"], "direction": "DESC"}` | N |
| "bottom N by X" | `{"order_by": ["X"], "direction": "ASC"}` | N |
| "highest X" | `{"order_by": ["X"], "direction": "DESC"}` | null |
| "ordered by X" | `{"order_by": ["X"], "direction": "ASC"}` | null |

### 11.8 Grain Derivation Examples

How to determine grain from dimensions.

| Dimensions Array | Grain |
|------------------|-------|
| `["<dim1>"]` | "one row per <dim1>" |
| `["<dim1>", "<dim2>"]` | "one row per <dim1>-<dim2> combination" |
| `[]` | "one row total" or "one row per <entity>" |

### 11.9 Follow-Up Query Examples

How to handle follow-up queries.

| Prior Query | Follow-Up | Action | Result |
|-------------|-----------|--------|--------|
| "revenue by region" | "what about by product too?" | Append dimension | dimensions: ["region", "product"] |
| "revenue by region" | "only for East" | Add filter | filters: [{"field": "region", "op": "=", "value": "East"}] |
| "revenue by region" | "order count instead" | Replace metric | metrics: [order_count] |
| "revenue for Jan" | "what about Feb?" | Update time | time: {...Feb dates...} |

### 11.10 Error Recovery Examples

How to phrase clarification questions.

| Error Type | Example Message |
|------------|-----------------|
| column_not_found | "I can't find `<col>` in this dataset. Did you mean `<alt1>` or `<alt2>`?" |
| ambiguous_metric | "Did you mean `<metric1>` or `<metric2>`?" |
| missing_time_range | "What time period would you like to analyze?" |
```

---

## CHECKLIST FOR NEW DOMAIN_MD FILES

When creating a new domain_md for a new dataset/agent, ensure ALL these sections are complete:

### Required Sections

| # | Section | Required Fields | Purpose |
|---|---------|-----------------|---------|
| 1 | Domain identity | `domain_key`, `description` | Agent identification |
| 2 | Time semantics | `supports_no_time_queries`, `default_time_column`, `default_time_rule`, `time_columns_by_entity` | Time handling |
| 3 | Listing rules | `listing_allows_empty_metrics`, `listing_default_limit` | Query defaults |
| 3.5 | Data Structure | Path extraction instructions, view naming patterns, flat vs nested, examples | File → View mapping |
| 4 | Core entities | `name`, `typical_grain`, `default_start_table_hint`, `view_examples`, `query_strategy`, `grain_examples` | Table discovery |
| 5 | Dimensions | `dimension_name`, `table`, `column`, `description`, `common_queries`, `synonyms`, `usage_notes` | Column mapping |
| 6 | Common filters | `default_filters`, `filter_patterns`, `popular_query_templates` | Query patterns |
| 7 | Metrics | `metric_name`, `definition`, `required_tables`, `example_sql` | Metric→SQL |
| 8 | Join conventions | `canonical_joins` (left_table, right_table, on, join_type, examples) | Join planning |
| 9 | Safety defaults | `default_limit`, `avoid_select_star`, `allow_cross_join` | Execution limits |
| 10 | Notes | Default interpretations, PM policies | Business rules |
| **11** | **Example Patterns** | All 10 subsections (11.1-11.10) | LLM training examples |

### Section 11 Subsections Checklist

| Subsection | Must Include | Minimum Examples |
|------------|--------------|------------------|
| 11.1 User Query → Query Spec | Full query_spec mapping | 3-5 queries |
| 11.2 Dimension Inference | Natural language → column mapping | 3-5 dimensions |
| 11.3 Metric Calculation | Definition → SQL with expansion | All metrics |
| 11.4 Filter Patterns | User language → filter JSON | 3-5 filters |
| 11.5 Join Detection | When joins are needed | 1-2 join examples |
| 11.6 UNION ALL | Multi-view aggregation (if applicable) | 1-2 examples |
| 11.7 Sorting/Limit | Top N, order by patterns | 3-4 patterns |
| 11.8 Grain Derivation | Dimensions → grain string | Cover all dimensions |
| 11.9 Follow-Up Queries | Append/replace/filter patterns | 4-5 examples |
| 11.10 Error Recovery | Clarification message templates | 2-3 error types |

---

## REFACTORING PLAN FOR SYSTEM PROMPTS

### Goal
Make system prompts data-agnostic by:
1. Removing all hardcoded examples (table names, columns, metrics, etc.)
2. Replacing with placeholders that reference domain_md sections
3. Adding instructions to extract examples from domain_md

### Files to Refactor

1. **decider.md**
   - Remove: 15+ hardcoded view/table names
   - Remove: 10+ specific column names (region, category, etc.)
   - Remove: 5+ metric definitions (revenue, order_count, etc.)
   - Remove: 5+ filter examples
   - Remove: SQL snippets
   - Add: "Refer to domain_md Section X for examples"

2. **nl_to_sql_planner.md**
   - Remove: Path examples (ECommerce/sample_sales_data.csv)
   - Remove: UNION ALL view names
   - Remove: Join table names
   - Remove: Metric SQL expressions
   - Add: Instructions to use query_spec.metrics and domain_md

3. **ask_user_clarification.md**
   - Mostly agnostic already
   - Minor: Remove specific query examples if any

4. **response_commentary.md**
   - Already agnostic (no changes needed)

### Placeholder Conventions

Use these placeholders in prompts (to be resolved from domain_md):

| Placeholder | Source |
|-------------|--------|
| `<table_from_domain_md>` | domain_md Section 4 (Core entities → view_examples) |
| `<column_from_domain_md>` | domain_md Section 5 (Dimensions → column) |
| `<metric_from_domain_md>` | domain_md Section 7 (Metrics → definition) |
| `<path_from_domain_md>` | domain_md Section 3.5 (Data Structure → path examples) |
| `<pattern_from_domain_md>` | domain_md Section 4 (Core entities → default_start_table_hint) |
| `<join_from_domain_md>` | domain_md Section 8 (Join conventions → canonical_joins) |
| `<example_from_domain_md>` | domain_md Section 11 (Example Patterns) |

---

## BACKWARD COMPATIBILITY

When refactoring:
1. Ensure existing agents (ecomm, ecommerce_advanced) continue to work
2. Add Section 11 to existing domain_md files
3. Test all 10 queries after each refactoring step
4. Commit after each successful test

---

## VERSION HISTORY

| Date | Version | Changes |
|------|---------|---------|
| 2024-12-30 | 1.0 | Initial inventory and template |

