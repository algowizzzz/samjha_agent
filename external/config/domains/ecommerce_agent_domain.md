# Domain: ecomm

## 1) Domain identity
- domain_key: ecomm
- description: ECommerce warehouse covering customers, inventory, and sales.

---

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
  - inventory: ""

---

## 3) Listing rules

- listing_allows_empty_metrics: true
- listing_default_limit: 50

---

## 3.5) Data Structure and Path Extraction

Data files are organized in a flat structure.

**Path Extraction Instructions:**
- Table names from Section 4 (default_start_table_hint) correspond to CSV files
- Path extraction: Construct paths as `{table_name}.csv`
- Example: `sample_sales_data` → `sample_sales_data.csv`
- No subfolders: paths are flat (filename only)

---

## 4) Core entities (hints only)

- primary_entities:
  - name: customers
    typical_grain: one row per customer
    default_start_table_hint: sample_customer_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique customer_id
      - Grain affects aggregation: customer-level metrics require grouping by customer_id
    grain_examples:
      - "list customers" → grain: one row per customer
      - "customer purchases" → grain: one row per customer (aggregated from sales)

  - name: products
    typical_grain: one row per product
    default_start_table_hint: sample_inventory_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique product_id or product_name
      - Grain affects aggregation: product-level metrics require grouping by product
    grain_examples:
      - "list products" → grain: one row per product
      - "top products" → grain: one row per product (aggregated from sales)

  - name: sales
    typical_grain: one row per order line
    default_start_table_hint: sample_sales_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique order_id + product combinations
      - Grain affects aggregation: sum(quantity) groups by product/region/category
    grain_examples:
      - "top products" → grain: one row per product (aggregated from order lines)
      - "revenue by region" → grain: one row per region (aggregated from order lines)
      - "sales by category" → grain: one row per category (aggregated from order lines)

---

## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: region
    table: sample_sales_data
    column: region
    description: Sales region (North, South, East, West)
    common_queries: ["revenue by region", "sales by region", "total by region"]
    synonyms: []
    usage_notes: "Use for geographic breakdowns. Verify column exists via inspect_table before SQL generation."

  - dimension_name: product
    table: sample_sales_data
    column: product
    description: Product name
    common_queries: ["top products", "products by sales", "revenue by product"]
    synonyms: ["item", "sku"]
    usage_notes: "Use for product-level analysis. Groups order lines by product name."

  - dimension_name: category
    table: sample_sales_data
    column: category
    description: Product category (e.g., Electronics, Furniture)
    common_queries: ["revenue by category", "sales by category", "products by category"]
    synonyms: []
    usage_notes: "Use for category-level aggregation. Verify column exists via inspect_table."

  - dimension_name: customer_id
    table: sample_sales_data
    column: customer_id
    description: Customer identifier
    common_queries: ["customer purchases", "sales by customer", "top customers"]
    synonyms: []
    usage_notes: "Use for customer-level analysis. May require join to customer_data for customer names."

  - dimension_name: order_date
    table: sample_sales_data
    column: order_date
    description: Date of order transaction
    common_queries: ["sales over time", "revenue by date", "daily sales"]
    synonyms: []
    usage_notes: "Use for time-based analysis. Can be grouped by day/month/year."

---

## 6) Common filters (Decider reference)

- default_filters:
  - active_products: "is_active = true" (if column exists in inventory table)
  - completed_orders: "order_status = 'completed'" (if column exists in sales table)
  
- filter_patterns:
  - by_region: "region IN ('North', 'South', 'East', 'West')" or "region = '...'"
  - by_category: "category = 'Electronics'" or "category IN (...)"
  - by_date_range: "order_date BETWEEN '2024-01-01' AND '2024-12-31'"
  - by_customer: "customer_id = 'C001'" or "customer_id IN (...)"
  - by_product: "product = 'Laptop'" or "product IN (...)"
  
- popular_query_templates:
  - "revenue by region": 
    dimensions: [region]
    filters: []
    grain: one row per region
    aggregation: sum(quantity * price) group by region
    
  - "top products": 
    dimensions: [product]
    filters: []
    grain: one row per product
    aggregation: sum(quantity) group by product order by sum(quantity) desc limit N
    
  - "sales by category": 
    dimensions: [category]
    filters: []
    grain: one row per category
    aggregation: sum(quantity * price) group by category
    
  - "revenue by region and category": 
    dimensions: [region, category]
    filters: []
    grain: one row per region-category combination
    aggregation: sum(quantity * price) group by region, category

---

## 7) Metric dictionary

- metrics:
  - metric_name: revenue
    definition: Sum of (quantity * price)
    required_tables: ["sample_sales_data"]
    usage_notes: "DEFAULT meaning of 'sales' when the user says 'sales' without specifying quantity vs revenue. Prefer revenue unless user explicitly requests quantity/order_count."

  - metric_name: order_count
    definition: Count of distinct order_id
    required_tables: ["sample_sales_data"]

  - metric_name: avg_order_value
    definition: revenue / order_count
    required_tables: ["sample_sales_data"]

  - metric_name: stock_value
    definition: Sum of (stock_quantity * unit_cost)
    required_tables: ["sample_inventory_data"]

---

## 8) Join conventions

- canonical_joins:
  - left_table: sample_sales_data
    right_table: sample_customer_data
    on: sample_sales_data.customer_id = sample_customer_data.customer_id
    join_type: left

  - left_table: sample_sales_data
    right_table: sample_inventory_data
    on: sample_sales_data.product = sample_inventory_data.product_name
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

- Time defaults are **optional**, not mandatory.
- If the user does not request time, use `no_time`.
- Investigation should always precede clarification.
- Default interpretation (PM policy): If user says "top products by sales" and doesn't specify, interpret "sales" as **revenue** (SUM(quantity * price)) and default to top 10 unless user specifies N.

---

## 11) Example Patterns (LLM Reference Examples)

This section provides concrete examples for the LLM to understand how to process queries for this dataset. **System prompts should reference this section instead of using hardcoded examples.**

### 11.1 User Query → Query Spec Examples

| User Query | metrics | dimensions | filters | sorting | limit | grain |
|------------|---------|------------|---------|---------|-------|-------|
| "top products by sales" | [{"name": "revenue", "definition": "Sum of (quantity * price)"}] | ["product"] | - | {"order_by": ["revenue"], "direction": "DESC"} | 10 | "one row per product" |
| "revenue by region" | [{"name": "revenue", "definition": "Sum of (quantity * price)"}] | ["region"] | - | - | - | "one row per region" |
| "sales by category" | [{"name": "revenue", "definition": "Sum of (quantity * price)"}] | ["category"] | - | - | - | "one row per category" |
| "revenue by region and category" | [{"name": "revenue", "definition": "Sum of (quantity * price)"}] | ["region", "category"] | - | - | - | "one row per region-category combination" |
| "order count by region" | [{"name": "order_count", "definition": "Count of distinct order_id"}] | ["region"] | - | - | - | "one row per region" |

### 11.2 Dimension Inference Examples

| User Language | Dimension Found | Column Used | Table |
|---------------|-----------------|-------------|-------|
| "by region" | region | region | sample_sales_data |
| "top products" | product | product | sample_sales_data |
| "by category" | category | category | sample_sales_data |
| "by customer" | customer_id | customer_id | sample_sales_data |
| "over time" / "by date" | order_date | order_date | sample_sales_data |

### 11.3 Metric Calculation Examples

| Metric Name | Definition (Natural) | SQL Expression |
|-------------|---------------------|----------------|
| revenue | "Sum of (quantity * price)" | `SUM(quantity * price) AS revenue` |
| order_count | "Count of distinct order_id" | `COUNT(DISTINCT order_id) AS order_count` |
| avg_order_value | "revenue / order_count" | `SUM(quantity * price) / COUNT(DISTINCT order_id) AS avg_order_value` |
| stock_value | "Sum of (stock_quantity * unit_cost)" | `SUM(stock_quantity * unit_cost) AS stock_value` |

**Metric Expansion Example:**
- Query: "average order value"
- `avg_order_value = revenue / order_count`
- `revenue = Sum of (quantity * price)` → `SUM(quantity * price)`
- `order_count = Count of distinct order_id` → `COUNT(DISTINCT order_id)`
- Final SQL: `SUM(quantity * price) / COUNT(DISTINCT order_id) AS avg_order_value`

### 11.4 Filter Pattern Examples

| User Language | Filter Generated |
|---------------|------------------|
| "only East" / "for East region" | `{"field": "region", "op": "=", "value": "East"}` |
| "Electronics only" | `{"field": "category", "op": "=", "value": "Electronics"}` |
| "in 2024" | `{"field": "order_date", "op": "BETWEEN", "start": "2024-01-01", "end": "2024-12-31"}` |
| "for customer C001" | `{"field": "customer_id", "op": "=", "value": "C001"}` |
| "Laptop only" | `{"field": "product", "op": "=", "value": "Laptop"}` |

### 11.5 Join Detection Examples

| Query Pattern | Start Table | Join Table | Join Required | Join Condition |
|---------------|-------------|------------|---------------|----------------|
| "sales with customer info" | sample_sales_data | sample_customer_data | Yes | `sample_sales_data.customer_id = sample_customer_data.customer_id` |
| "sales with inventory cost" | sample_sales_data | sample_inventory_data | Yes | `sample_sales_data.product = sample_inventory_data.product_name` |
| "revenue by region" | sample_sales_data | - | No | - |
| "top products" | sample_sales_data | - | No | - |

### 11.6 Sorting and Limit Examples

| User Language | sorting | limit |
|---------------|---------|-------|
| "top 5 products" | `{"order_by": ["revenue"], "direction": "DESC"}` | 5 |
| "top 10 by revenue" | `{"order_by": ["revenue"], "direction": "DESC"}` | 10 |
| "lowest 3 by sales" | `{"order_by": ["revenue"], "direction": "ASC"}` | 3 |
| "ordered by date" | `{"order_by": ["order_date"], "direction": "ASC"}` | null |

### 11.7 Grain Derivation Examples

| Dimensions Array | Grain |
|------------------|-------|
| `["region"]` | "one row per region" |
| `["product"]` | "one row per product" |
| `["category"]` | "one row per category" |
| `["region", "category"]` | "one row per region-category combination" |
| `["order_date"]` | "one row per order_date" |
| `[]` | "one row total" |

### 11.8 Follow-Up Query Examples

| Prior Query | Follow-Up | Action | Result |
|-------------|-----------|--------|--------|
| "revenue by region" | "what about by product too?" | Append dimension | dimensions: ["region", "product"] |
| "revenue by region" | "only for East" | Add filter, clear dimension | filters: [region=East], dimensions: [] |
| "revenue by region" | "order count instead" | Replace metric | metrics: [order_count] |
| "sales in 2024" | "what about just January?" | Update time | time: {order_date BETWEEN 2024-01-01 AND 2024-01-31} |

### 11.9 Error Recovery Examples

| Error Type | Example Message |
|------------|-----------------|
| column_not_found | "I can't find `promo_code` in this dataset. Did you mean `product` or `category`?" |
| ambiguous_metric | "By 'sales', did you mean revenue (total value) or order_count (number of orders)?" |

### 11.10 Table/View Reference Examples

| Entity | Table Name | Path |
|--------|------------|------|
| sales | sample_sales_data | sample_sales_data.csv |
| customers | sample_customer_data | sample_customer_data.csv |
| inventory | sample_inventory_data | sample_inventory_data.csv |
