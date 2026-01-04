# Complete Domain Configuration Guide

> **Your Step-by-Step Guide to Creating Domain Configurations for Natural Language Data Queries**

This guide helps you create domain configuration files that enable users to ask questions about your data in plain English. No SQL knowledge required for your users—the system translates natural language into queries automatically.

---

## Quick Start

**What is a Domain Configuration?**
A `.md` file that teaches the system about your data: what tables you have, how they relate, what metrics matter, and how to answer common questions.

**Where does it go?**
Upload via the Admin UI when creating an agent, or place in `external/config/domains/<your_domain>_domain.md`

**How long does it take?**
- Simple dataset (1-3 tables): ~30 minutes
- Medium dataset (4-7 tables): ~1-2 hours
- Complex dataset (8+ tables with many joins): ~2-4 hours

---

## Table of Contents

1. [Domain Identity](#section-1-domain-identity)
2. [Time Semantics](#section-2-time-semantics)
3. [Listing Rules](#section-3-listing-rules)
4. [Data Structure (Optional)](#section-35-data-structure-optional)
5. [Core Entities](#section-4-core-entities)
6. [Dimensions Dictionary](#section-5-dimensions-dictionary)
7. [Common Filters](#section-6-common-filters)
8. [Metric Dictionary](#section-7-metric-dictionary)
9. [Join Conventions](#section-8-join-conventions)
10. [Safety Defaults](#section-9-safety-defaults)
11. [Notes](#section-10-notes)
12. [LLM Example Patterns](#section-11-llm-example-patterns-critical) ⭐ **Critical for AI behavior**
13. [Complete Template](#complete-template)
14. [Checklist](#final-checklist)

---

## Section 1: Domain Identity

**What it does:** Names and describes your domain. This is the "name tag" for your data.

### Fill This Out

```markdown
## 1) Domain identity
- domain_key: _______________
- description: _______________________________________________
```

### Guidelines

| Field | Rules | Examples |
|-------|-------|----------|
| `domain_key` | Lowercase, 3-20 chars, underscores OK | `ecomm`, `retail`, `finance`, `hr_analytics` |
| `description` | 1-3 sentences about your data | `"E-commerce warehouse with customers, products, and sales transactions."` |

### Example

```markdown
## 1) Domain identity
- domain_key: retail_sales
- description: Retail store sales data covering stores, products, daily transactions, and inventory levels across all US locations.
```

---

## Section 2: Time Semantics

**What it does:** Tells the system how to handle time-based queries. Should "recent sales" mean last 7 days or last 30? This section decides.

### Fill This Out

```markdown
## 2) Time semantics (Decider reference)

- supports_no_time_queries: true/false
- apply_default_time_rule_when: explicit_or_implied_time_only / always / never
- default_time_column: _______________
- default_time_rule: last_n_days / as_of_latest / no_time
- default_time_n_days: ___

### Time columns by entity
- time_columns_by_entity:
  - entity1: column_name
  - entity2: column_name
  - entity3: ""
```

### Quick Decision Guide

| Question | If Yes | If No |
|----------|--------|-------|
| Can users ask queries without time filters? (e.g., "list all products") | `supports_no_time_queries: true` | `supports_no_time_queries: false` |
| Apply time filter only when user mentions time? | `apply_default_time_rule_when: explicit_or_implied_time_only` | Use `always` or `never` |

### Default Time Rule Options

| Rule | When to Use | Example Query |
|------|-------------|---------------|
| `last_n_days` | Most common. Recent data matters. | "recent sales" → last 30 days |
| `as_of_latest` | Point-in-time snapshots | "current inventory" → latest available date |
| `no_time` | Static data, no date filtering | Reference tables, product catalogs |

### Example

```markdown
## 2) Time semantics (Decider reference)

- supports_no_time_queries: true
- apply_default_time_rule_when: explicit_or_implied_time_only
- default_time_column: order_date
- default_time_rule: last_n_days
- default_time_n_days: 30

### Time columns by entity
- time_columns_by_entity:
  - sales: order_date
  - customers: signup_date
  - products: ""
  - inventory: snapshot_date
```

---

## Section 3: Listing Rules

**What it does:** Controls simple list queries like "show me all customers" or "list products."

### Fill This Out

```markdown
## 3) Listing rules

- listing_allows_empty_metrics: true/false
- listing_default_limit: ___
```

### Guidelines

| Field | What It Means | Recommended |
|-------|---------------|-------------|
| `listing_allows_empty_metrics` | Can users say "list customers" without specifying a metric like revenue? | `true` for exploratory data, `false` for strict reporting |
| `listing_default_limit` | Max rows returned for list queries | 50-100 for most cases |

### Example

```markdown
## 3) Listing rules

- listing_allows_empty_metrics: true
- listing_default_limit: 50
```

---

## Section 3.5: Data Structure (Optional)

**What it does:** Explains special data organization (like monthly snapshots, partitioned folders, or view naming patterns). Skip if your data is simple tables.

### When You Need This

- Data split across multiple files/folders (e.g., monthly snapshots)
- Views have naming patterns (e.g., `jan2024_sales`, `feb2024_sales`)
- Need UNION ALL queries for trend analysis

### Example (Monthly Snapshots)

```markdown
## 3.5) Data Structure and View Naming

Data is organized in monthly subfolders. Each file becomes a view:

**Examples:**
- `jan2024/sales.csv` → view: `jan2024_sales`
- `feb2024/sales.csv` → view: `feb2024_sales`

**Querying strategy:**
- Single month: Use specific view (e.g., `SELECT * FROM jan2024_sales`)
- Multi-month trends: UNION ALL all matching views

**Date column:**
All views include `report_date` matching the folder date.
```

---

## Section 4: Core Entities

**What it does:** Defines the main "things" in your data (customers, products, orders, etc.) and their **grain** (what one row represents).

### Understanding Grain (Critical!)

**Grain = What one row in your table represents**

| Table | Grain | What This Means |
|-------|-------|-----------------|
| `customers` | one row per customer | Each row = one customer (C001, C002...) |
| `products` | one row per product | Each row = one product (Laptop, Chair...) |
| `order_lines` | one row per order line | One order can have multiple rows (Order123 + Laptop, Order123 + Chair) |
| `daily_sales` | one row per store per day | Aggregate, not transactional |

### Fill This Out (Repeat for Each Entity)

```markdown
## 4) Core entities (hints only)

- primary_entities:
  - name: _______________
    typical_grain: one row per _______________
    default_start_table_hint: _______________
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique _______________
    grain_examples:
      - "list [entity]" → grain: one row per ___
      - "[entity] [metric]" → grain: one row per ___ (aggregated from ___)
```

### Example

```markdown
## 4) Core entities (hints only)

- primary_entities:
  - name: customers
    typical_grain: one row per customer
    default_start_table_hint: customer_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique customer_id
    grain_examples:
      - "list customers" → grain: one row per customer
      - "customer revenue" → grain: one row per customer (aggregated from sales)

  - name: sales
    typical_grain: one row per order line
    default_start_table_hint: sales_transactions
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique order_id + product_id
    grain_examples:
      - "top products" → grain: one row per product (aggregated from order lines)
      - "revenue by region" → grain: one row per region (aggregated from order lines)
```

---

## Section 5: Dimensions Dictionary

**What it does:** Lists all columns users can group by or break down data. Think: "revenue **by region**" - `region` is a dimension.

### What Makes a Good Dimension?

✅ **Good dimensions:** region, category, product_name, customer_segment, order_date, store_id
❌ **Not dimensions:** quantity, price, revenue (these are metrics, not for grouping)

### Fill This Out (Repeat for Each Dimension)

```markdown
## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: _______________
    table: _______________
    column: _______________
    description: _______________
    common_queries: ["___ by ___", "___ by ___"]
    synonyms: ["___", "___"]
    usage_notes: "_______________"
```

### Example

```markdown
## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: region
    table: sales_data
    column: region
    description: Sales region (North, South, East, West)
    common_queries: ["revenue by region", "sales by region"]
    synonyms: ["area", "territory"]
    usage_notes: "Primary geographic breakdown for all sales analysis."

  - dimension_name: category
    table: sales_data
    column: product_category
    description: Product category (Electronics, Furniture, Clothing)
    common_queries: ["revenue by category", "top categories"]
    synonyms: ["product type", "dept"]
    usage_notes: "Use for category-level aggregation."

  - dimension_name: customer_tier
    table: customer_data
    column: tier
    description: Customer tier (Premium, Standard, Basic)
    common_queries: ["revenue by customer tier", "avg order by tier"]
    synonyms: ["tier", "customer segment", "loyalty level"]
    usage_notes: "Requires JOIN from sales to customer table via customer_id."
```

---

## Section 6: Common Filters

**What it does:** Defines default filters (auto-applied) and filter patterns users commonly request.

### Fill This Out

```markdown
## 6) Common filters (Decider reference)

- default_filters:
  - filter_name: "SQL condition" (note)
  
- filter_patterns:
  - by_dimension: "column = 'value'" or "column IN (...)"
  
- popular_query_templates:
  - "query pattern": 
    dimensions: [dim1, dim2]
    filters: []
    grain: one row per ___
    aggregation: sum(metric) group by [dimensions]
```

### Example

```markdown
## 6) Common filters (Decider reference)

- default_filters:
  - active_only: "is_active = true" (if column exists)
  - completed_orders: "status = 'completed'" (exclude cancelled)
  
- filter_patterns:
  - by_region: "region = 'North'" or "region IN ('North', 'South')"
  - by_category: "category = 'Electronics'" or "category IN (...)"
  - by_date_range: "order_date BETWEEN '2024-01-01' AND '2024-03-31'"
  
- popular_query_templates:
  - "top products by revenue": 
    dimensions: [product]
    filters: []
    grain: one row per product
    aggregation: sum(quantity * price) group by product order by revenue desc limit 10
    
  - "revenue by region and category": 
    dimensions: [region, category]
    filters: []
    grain: one row per region-category combination
    aggregation: sum(quantity * price) group by region, category
```

---

## Section 7: Metric Dictionary

**What it does:** Defines how to calculate business metrics. This is your "recipe book" for numbers.

### Common Metric Patterns

| Pattern | Formula | Example |
|---------|---------|---------|
| **Sum** | `Sum of (column)` | `revenue = Sum of (quantity * price)` |
| **Count** | `Count of distinct column` | `order_count = Count of distinct order_id` |
| **Average** | `Average of (column)` | `avg_price = Average of (price)` |
| **Ratio** | `metric1 / metric2` | `avg_order_value = revenue / order_count` |
| **Difference** | `metric1 - metric2` | `profit = revenue - cost` |

### Fill This Out

```markdown
## 7) Metric dictionary

- metrics:
  - metric_name: _______________
    definition: _______________
    required_tables: ["___"]
```

### Example

```markdown
## 7) Metric dictionary

- metrics:
  - metric_name: revenue
    definition: Sum of (quantity * price)
    required_tables: ["sales_data"]

  - metric_name: order_count
    definition: Count of distinct order_id
    required_tables: ["sales_data"]

  - metric_name: avg_order_value
    definition: revenue / order_count
    required_tables: ["sales_data"]

  - metric_name: profit
    definition: Sum of (revenue) - Sum of (cost)
    required_tables: ["sales_data"]

  - metric_name: stock_value
    definition: Sum of (quantity * unit_cost)
    required_tables: ["inventory_data"]
```

---

## Section 8: Join Conventions

**What it does:** Tells the system how tables connect to each other (foreign key relationships).

### Fill This Out

```markdown
## 8) Join conventions

- canonical_joins:
  - left_table: _______________
    right_table: _______________
    on: table1.column = table2.column
    join_type: left / inner / right

- forbidden_joins: []
```

### Join Types Explained

| Type | When to Use | What Happens |
|------|-------------|--------------|
| `left` | Most common. Keep all rows from left table. | All sales shown, even if customer info missing |
| `inner` | Only want matched rows | Only sales with valid customer records |
| `right` | Rarely used | Keep all rows from right table |

### Example

```markdown
## 8) Join conventions

- canonical_joins:
  - left_table: sales_data
    right_table: customer_data
    on: sales_data.customer_id = customer_data.customer_id
    join_type: left

  - left_table: sales_data
    right_table: product_data
    on: sales_data.product_id = product_data.product_id
    join_type: left

  - left_table: sales_data
    right_table: store_data
    on: sales_data.store_id = store_data.store_id
    join_type: left

- forbidden_joins: []
```

---

## Section 9: Safety Defaults

**What it does:** Prevents runaway queries from returning millions of rows or crashing the system.

### Fill This Out

```markdown
## 9) Safety defaults (Executor reference)

- performance_guardrails:
  - default_limit: ___
  - avoid_select_star: true/false
  - allow_cross_join: true/false
```

### Recommended Settings by Dataset Size

| Dataset Size | default_limit | avoid_select_star | allow_cross_join |
|--------------|---------------|-------------------|------------------|
| Small (< 10K rows) | 100 | false | false |
| Medium (10K - 1M) | 50 | true | false |
| Large (> 1M rows) | 25 | true | false |

### Example

```markdown
## 9) Safety defaults (Executor reference)

- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false
```

---

## Section 10: Notes

**What it does:** Document any special rules, limitations, or important information about your data.

### Common Things to Document

- **Time behavior:** "Time defaults are optional, not mandatory."
- **Business rules:** "Revenue excludes refunded orders."
- **Data quality:** "Some customer records have NULL email addresses."
- **Limitations:** "Historical data only available from 2020 onwards."
- **Default interpretations:** "When user says 'sales', interpret as revenue."

### Example

```markdown
## 10) Notes

- Time defaults are **optional**, not mandatory.
- If the user does not specify time, use `no_time` (all available data).
- Investigation should always precede clarification.
- "Sales" without specification means revenue (quantity * price).
- Customer tiers require join to customer table.
```

---

## Section 11: LLM Example Patterns (CRITICAL)

**What it does:** This is the most important section for AI behavior! It provides concrete examples that teach the AI when to proceed vs. ask for clarification.

### Why This Section Matters

Without this section, the AI might:
- Ask too many clarifying questions for simple queries
- Misinterpret common patterns like "top products"
- Fail to infer obvious defaults

With this section, the AI:
- Handles common queries confidently
- Uses your business defaults automatically
- Only asks questions when genuinely needed

---

### 11.0 PROCEED vs ASK_USER Policy

**Copy this exactly and customize the examples:**

```markdown
## 11) Example Patterns (LLM Reference Examples)

### 11.0 PROCEED vs ASK_USER Policy (CRITICAL)

**PROCEED without clarification when:**
1. Query matches an example in Section 11.1
2. All required fields can be resolved from Sections 5-7
3. Join is needed but defined in Section 8 → Fill joins array and PROCEED
4. Query uses standard patterns: "top N", "by X", "over time"

**ASK_USER only when:**
1. Query references a column/entity NOT in this domain
2. Query is genuinely ambiguous with no default
3. Multiple interpretations with significantly different results

**Default behaviors (use these, don't ask):**
- "top products" / "best products" → revenue metric, DESC sort, limit 10
- "sales" without metric specified → revenue = SUM(quantity * price)
- "lowest" / "bottom" → ASC sort
- No limit specified for "top N" → default to 10
```

---

### 11.1 Query to Spec Examples

**This is your "cheat sheet" for the AI. Add 10-15 examples covering your common queries:**

```markdown
### 11.1 User Query → Query Spec Examples

| User Query | start_table | metrics | dimensions | filters | sorting | limit |
|------------|-------------|---------|------------|---------|---------|-------|
| "top products by sales" | sales_data | [revenue] | [product] | - | DESC by revenue | 10 |
| "revenue by region" | sales_data | [revenue] | [region] | - | - | - |
| "sales by category" | sales_data | [revenue] | [category] | - | - | - |
| "top 5 customers" | sales_data | [revenue] | [customer_id] | - | DESC by revenue | 5 |
| "sales in January" | sales_data | [revenue] | [] | [order_date in Jan] | - | - |
| "lowest selling products" | sales_data | [revenue] | [product] | - | ASC by revenue | 10 |
| "order count by region" | sales_data | [order_count] | [region] | - | - | - |
| "average order value by tier" | sales_data | [avg_order_value] | [customer_tier] | - | - | - |
```

---

### 11.2-11.4 Inference Examples

```markdown
### 11.2 Dimension Inference Examples

| User Language | Dimension | Column | Table |
|---------------|-----------|--------|-------|
| "by region" | region | region | sales_data |
| "top products" | product | product_name | sales_data |
| "by category" | category | category | sales_data |
| "by customer tier" | customer_tier | tier | customer_data |
| "over time" / "by month" | order_date | order_date | sales_data |

### 11.3 Metric Calculation Examples

| Metric Name | Definition | SQL Expression |
|-------------|-----------|----------------|
| revenue | Sum of (quantity * price) | `SUM(quantity * price)` |
| order_count | Count of distinct order_id | `COUNT(DISTINCT order_id)` |
| avg_order_value | revenue / order_count | `SUM(quantity * price) / COUNT(DISTINCT order_id)` |

### 11.4 Filter Pattern Examples

| User Language | Filter Generated |
|---------------|------------------|
| "only East" / "for East region" | `region = 'East'` |
| "Electronics only" | `category = 'Electronics'` |
| "in January" | `order_date BETWEEN '2024-01-01' AND '2024-01-31'` |
| "last 30 days" | `order_date >= CURRENT_DATE - 30` |
```

---

### 11.5 Join Detection Examples

```markdown
### 11.5 Join Detection Examples

| Query Pattern | Start Table | Needs Join To | Join Condition |
|---------------|-------------|---------------|----------------|
| "revenue by customer tier" | sales_data | customer_data | customer_id = customer_id |
| "sales with product details" | sales_data | product_data | product_id = product_id |
| "revenue by region" | sales_data | (same table) | No join needed |

**CRITICAL:** When join is needed, populate the `joins` array and PROCEED. Do NOT ask user.

Example output for "revenue by customer tier":
```json
{
  "start_table": "sales_data",
  "dimensions": ["customer_tier"],
  "joins": [
    {
      "left_table": "sales_data",
      "right_table": "customer_data", 
      "on": "sales_data.customer_id = customer_data.customer_id",
      "join_type": "LEFT"
    }
  ]
}
```
```

---

### 11.6-11.8 More Examples

```markdown
### 11.6 Sorting and Limit Examples

| User Language | Sorting | Limit |
|---------------|---------|-------|
| "top 5 products" | DESC by revenue | 5 |
| "top 10 by revenue" | DESC by revenue | 10 |
| "lowest 3 by sales" | ASC by revenue | 3 |
| "ordered by date" | ASC by order_date | null |
| "best performing" | DESC by revenue | 10 |

### 11.7 Grain Derivation Examples

| Dimensions Array | Grain |
|------------------|-------|
| `["region"]` | "one row per region" |
| `["product"]` | "one row per product" |
| `["region", "category"]` | "one row per region-category combination" |
| `[]` (no dimensions) | "one row total" |

### 11.8 Follow-Up Query Examples

| Prior Query | Follow-Up | Action |
|-------------|-----------|--------|
| "revenue by region" | "also by category" | Add dimension: ["region", "category"] |
| "revenue by region" | "only East" | Add filter, keep dimension |
| "top products" | "now by category" | Replace dimension: ["category"] |
| "sales for Jan" | "what about Feb?" | Update time filter |
```

---

### 11.9 Error Recovery

```markdown
### 11.9 Error Recovery Examples

| Error Type | Action |
|------------|--------|
| Column not found | ASK_USER with suggestions |
| Ambiguous metric | Use default (revenue) |
| Join required | PROCEED (fill joins array from Section 8) |
| Unknown entity | ASK_USER |
```

---

## Complete Template

Copy this entire template and fill in your values:

```markdown
# Domain: [YOUR_DOMAIN_KEY]

## 1) Domain identity
- domain_key: [your_domain_key]
- description: [1-3 sentence description]

---

## 2) Time semantics (Decider reference)

- supports_no_time_queries: true
- apply_default_time_rule_when: explicit_or_implied_time_only
- default_time_column: [your_date_column]
- default_time_rule: last_n_days
- default_time_n_days: 30

### Time columns by entity
- time_columns_by_entity:
  - [entity1]: [date_column]
  - [entity2]: [date_column]
  - [entity3]: ""

---

## 3) Listing rules

- listing_allows_empty_metrics: true
- listing_default_limit: 50

---

## 4) Core entities (hints only)

- primary_entities:
  - name: [entity_name]
    typical_grain: one row per [what]
    default_start_table_hint: [table_name]
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique [id_column]
    grain_examples:
      - "list [entity]" → grain: one row per [what]

---

## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: [dim_name]
    table: [table_name]
    column: [column_name]
    description: [description]
    common_queries: ["query1", "query2"]
    synonyms: []
    usage_notes: "[notes]"

---

## 6) Common filters (Decider reference)

- default_filters: []
  
- filter_patterns:
  - by_[dimension]: "[column] = 'value'"
  
- popular_query_templates:
  - "[query pattern]": 
    dimensions: [[dim1]]
    filters: []
    grain: one row per [what]
    aggregation: [sql pattern]

---

## 7) Metric dictionary

- metrics:
  - metric_name: [metric]
    definition: [calculation]
    required_tables: ["[table]"]

---

## 8) Join conventions

- canonical_joins:
  - left_table: [table1]
    right_table: [table2]
    on: [table1].[column] = [table2].[column]
    join_type: left

- forbidden_joins: []

---

## 9) Safety defaults (Executor reference)

- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false

---

## 10) Notes

- [Your important notes here]

---

## 11) Example Patterns (LLM Reference Examples)

### 11.0 PROCEED vs ASK_USER Policy (CRITICAL)

**PROCEED without clarification when:**
1. Query matches an example in Section 11.1
2. All required fields can be resolved from Sections 5-7
3. Join is needed but defined in Section 8

**ASK_USER only when:**
1. Query references something NOT in this domain
2. Query is genuinely ambiguous

**Default behaviors:**
- "top products" → revenue, DESC, limit 10
- "sales" → revenue = SUM(quantity * price)

### 11.1 User Query → Query Spec Examples

| User Query | start_table | metrics | dimensions | sorting | limit |
|------------|-------------|---------|------------|---------|-------|
| "top products" | [table] | [revenue] | [product] | DESC | 10 |
| "revenue by region" | [table] | [revenue] | [region] | - | - |
| [add more...] |

### 11.2 Dimension Inference Examples

| User Language | Dimension | Column |
|---------------|-----------|--------|
| "by region" | region | region |
| [add more...] |

### 11.3 Metric Calculation Examples

| Metric | Definition | SQL |
|--------|-----------|-----|
| revenue | Sum of (quantity * price) | SUM(quantity * price) |
| [add more...] |

### 11.4 Filter Pattern Examples

| User Language | Filter |
|---------------|--------|
| "only East" | region = 'East' |
| [add more...] |

### 11.5 Join Detection Examples

| Query | Needs Join | Condition |
|-------|-----------|-----------|
| "by customer tier" | Yes → customer_data | customer_id = customer_id |
| [add more...] |

### 11.6 Sorting and Limit Examples

| User Language | Sorting | Limit |
|---------------|---------|-------|
| "top 5" | DESC | 5 |
| [add more...] |

### 11.7 Grain Derivation Examples

| Dimensions | Grain |
|------------|-------|
| ["region"] | one row per region |
| [add more...] |
```

---

## Final Checklist

Before uploading your domain configuration, verify:

### Required Sections
- [ ] **Section 1:** Domain key and description
- [ ] **Section 2:** Time semantics configured
- [ ] **Section 3:** Listing rules set
- [ ] **Section 4:** 3-5 core entities with grain definitions
- [ ] **Section 5:** 5-10 dimensions documented
- [ ] **Section 6:** Common filters and query templates
- [ ] **Section 7:** 5-10 metrics with clear definitions
- [ ] **Section 8:** Join conventions documented
- [ ] **Section 9:** Safety defaults configured
- [ ] **Section 10:** Important notes added

### Critical for AI Behavior
- [ ] **Section 11.0:** PROCEED vs ASK_USER policy defined
- [ ] **Section 11.1:** 10-15 query-to-spec examples
- [ ] **Section 11.2-11.4:** Dimension, metric, filter inference examples
- [ ] **Section 11.5:** Join detection examples with output format
- [ ] **Section 11.6-11.7:** Sorting, limit, and grain examples

### Data Accuracy
- [ ] Table names match your actual data files
- [ ] Column names are exact (case-sensitive!)
- [ ] Grain definitions are accurate
- [ ] Join conditions use correct columns

### Testing
- [ ] Tested with 5 simple queries
- [ ] Tested with 3 queries requiring joins
- [ ] Tested "top N" style queries
- [ ] Tested time-based queries

---

## Need Help?

1. **Review existing examples:** Check `external/config/domains/ecommerce_advanced_domain.md` for a comprehensive example
2. **Test incrementally:** Add a few sections, test, then add more
3. **Check column names:** Use `inspect_table` to verify your actual schema
4. **Iterate:** Domain files are living documents—update as you discover new patterns

---

**Happy querying! 🚀**

