# Domain: ecommerce_advanced

## 1) Domain identity
- domain_key: ecommerce_advanced
- description: ECommerce warehouse with monthly snapshots organized by date subfolders. Data includes customers, inventory, and sales with report_date column indicating the snapshot month.

---

## 2) Time semantics (Decider reference)

- supports_no_time_queries: true
- apply_default_time_rule_when: explicit_or_implied_time_only

- default_time_column: report_date
- default_time_rule: last_n_days
- default_time_n_days: 30

### Time columns by entity
- time_columns_by_entity:
  - sales: report_date
  - customers: report_date
  - products: report_date
  - inventory: report_date

---

## 3) Listing rules

- listing_allows_empty_metrics: true
- listing_default_limit: 50

---

## 3.5) Data Structure and View Naming

Data is organized in monthly subfolders. Each file becomes a view with the pattern:
`{date}_{entity}_{date}`

**Examples:**
- `jan012024/sales_jan012024.csv` → view: `jan012024_sales_jan012024`
- `feb012024/customer_feb012024.csv` → view: `feb012024_customer_feb012024`
- `mar012024/inventory_mar012024.csv` → view: `mar012024_inventory_mar012024`

**View naming patterns:**
- Sales views: `{date}_sales_{date}` (e.g., `jan012024_sales_jan012024`)
- Customer views: `{date}_customer_{date}` (e.g., `jan012024_customer_jan012024`)
- Inventory views: `{date}_inventory_{date}` (e.g., `jan012024_inventory_jan012024`)

**Date column:**
All views include a `report_date` column matching the folder date:
- `jan012024/` → `report_date = 2024-01-01`
- `feb012024/` → `report_date = 2024-02-01`
- `mar012024/` → `report_date = 2024-03-01`

**Querying strategy:**
- **Single month queries**: Use specific view (faster, simpler)
  - Example: `SELECT * FROM jan012024_sales_jan012024 WHERE ...`
  
- **Multi-month/trend queries**: UNION ALL all views matching pattern
  - Example: `SELECT * FROM jan012024_sales_jan012024 UNION ALL SELECT * FROM feb012024_sales_jan012024 UNION ALL SELECT * FROM mar012024_sales_mar012024`
  - All views have identical schema, so UNION ALL works directly
  - After UNION, filter/group by `report_date` column normally
  
- **Date filtering**: Use `report_date` column directly (e.g., `WHERE report_date = '2024-01-01'` or `WHERE report_date BETWEEN '2024-01-01' AND '2024-03-01'`)

**Path Extraction Instructions:**
When extracting file paths for investigation_plan steps (inspect_table, preview_rows):
- Domain_md examples use format: `file_path → view_name`
- Extract the LEFT side (before →) for file paths
- Example: `jan012024/sales_jan012024.csv → view: jan012024_sales_jan012024`
  - Extract: `jan012024/sales_jan012024.csv` (complete path with subfolder)
  - Do NOT extract just the filename or the view name

---

## 4) Core entities (hints only)

- primary_entities:
  - name: customers
    typical_grain: one row per customer
    default_start_table_hint: *_customer_*
    view_naming_pattern: {date}_customer_{date}
    view_examples:
      - jan012024_customer_jan012024 (January 2024 customer snapshot)
      - feb012024_customer_feb012024 (February 2024 customer snapshot)
      - mar012024_customer_mar012024 (March 2024 customer snapshot)
    query_strategy:
      - Single month: Use specific view (e.g., jan012024_customer_jan012024)
      - Multi-month: UNION ALL all views matching *_customer_* pattern
      - All views have same schema, just UNION ALL directly
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique customer_id
      - Grain affects aggregation: customer-level metrics require grouping by customer_id
    grain_examples:
      - "list customers" → grain: one row per customer
      - "customer purchases" → grain: one row per customer (aggregated from sales)

  - name: products
    typical_grain: one row per product
    default_start_table_hint: *_inventory_*
    view_naming_pattern: {date}_inventory_{date}
    view_examples:
      - jan012024_inventory_jan012024 (January 2024 inventory snapshot)
      - feb012024_inventory_feb012024 (February 2024 inventory snapshot)
      - mar012024_inventory_mar012024 (March 2024 inventory snapshot)
    query_strategy:
      - Single month: Use specific view (e.g., jan012024_inventory_jan012024)
      - Multi-month: UNION ALL all views matching *_inventory_* pattern
      - All views have same schema, just UNION ALL directly
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique product_id or product_name
      - Grain affects aggregation: product-level metrics require grouping by product
    grain_examples:
      - "list products" → grain: one row per product
      - "top products" → grain: one row per product (aggregated from sales)

  - name: sales
    typical_grain: one row per order line
    default_start_table_hint: *_sales_*
    view_naming_pattern: {date}_sales_{date}
    view_examples:
      - jan012024_sales_jan012024 (January 2024 sales data)
      - feb012024_sales_jan012024 (February 2024 sales data)
      - mar012024_sales_mar012024 (March 2024 sales data)
    query_strategy:
      - Single month: Use specific view (e.g., jan012024_sales_jan012024)
      - Multi-month/trend: UNION ALL all views matching *_sales_* pattern
      - Example multi-month query:
        ```sql
        SELECT report_date, SUM(quantity * price) as revenue
        FROM (
          SELECT * FROM jan012024_sales_jan012024
          UNION ALL SELECT * FROM feb012024_sales_jan012024
          UNION ALL SELECT * FROM mar012024_sales_mar012024
        ) GROUP BY report_date
        ```
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique order_id + product combinations
      - Grain affects aggregation: sum(quantity) groups by product/region/category
    grain_examples:
      - "top products" → grain: one row per product (aggregated from order lines)
      - "revenue by region" → grain: one row per region (aggregated from order lines)
      - "sales by category" → grain: one row per category (aggregated from order lines)
      - "sales over time" → grain: one row per report_date (aggregated from order lines, requires UNION)

---

## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: region
    table: *_sales_*
    column: region
    description: Sales region (North, South, East, West)
    common_queries: ["revenue by region", "sales by region", "total by region"]
    synonyms: []
    usage_notes: "Use for geographic breakdowns. For multi-month queries, UNION all *_sales_* views first, then group by region. Verify column exists via inspect_table before SQL generation."

  - dimension_name: product
    table: *_sales_*
    column: product
    description: Product name
    common_queries: ["top products", "products by sales", "revenue by product"]
    synonyms: ["item", "sku"]
    usage_notes: "Use for product-level analysis. Groups order lines by product name. For multi-month queries, UNION all *_sales_* views first."

  - dimension_name: category
    table: *_sales_*
    column: category
    description: Product category (e.g., Electronics, Furniture)
    common_queries: ["revenue by category", "sales by category", "products by category"]
    synonyms: []
    usage_notes: "Use for category-level aggregation. For multi-month queries, UNION all *_sales_* views first. Verify column exists via inspect_table."

  - dimension_name: customer_id
    table: *_sales_*
    column: customer_id
    description: Customer identifier
    common_queries: ["customer purchases", "sales by customer", "top customers"]
    synonyms: []
    usage_notes: "Use for customer-level analysis. For multi-month queries, UNION all *_sales_* views first. May require join to customer views for customer names."

  - dimension_name: customer_tier
    table: *_customer_*
    column: customer_tier
    description: Customer tier classification (e.g., Premium, Standard, Basic)
    common_queries: ["average order value by customer tier", "revenue by customer tier", "sales by tier"]
    synonyms: ["tier", "customer segment"]
    usage_notes: "Requires join from sales views to customer views using customer_id. For multi-month queries, UNION all *_sales_* views first, then join to customer views. See Section 8 for canonical join pattern."

  - dimension_name: report_date
    table: *_sales_*
    column: report_date
    description: Snapshot date for monthly data (YYYY-MM-DD format). Date values correspond to folder names: 2024-01-01 (jan012024), 2024-02-01 (feb012024), 2024-03-01 (mar012024), etc.
    common_queries: ["sales over time", "revenue by month", "trend analysis", "sales for past N months"]
    synonyms: ["month", "period", "snapshot_date"]
    usage_notes: "Use for time-based analysis. When querying multiple months, UNION all *_sales_* views first, then group by report_date. Can be grouped by day/month/year or filtered using standard date functions (BETWEEN, >=, <=)."

---

## 6) Common filters (Decider reference)

- default_filters:
  - active_products: "is_active = true" (if column exists in inventory views)
  - completed_orders: "order_status = 'completed'" (if column exists in sales views)
  
- filter_patterns:
  - by_region: "region IN ('North', 'South', 'East', 'West')" or "region = '...'"
  - by_category: "category = 'Electronics'" or "category IN (...)"
  - by_date_single_month: "report_date = '2024-01-01'" (for single month queries, use specific view)
  - by_date_range: "report_date BETWEEN '2024-01-01' AND '2024-03-01'" (for multi-month queries, UNION views first)
  - by_date_multiple_months: "report_date IN ('2024-01-01', '2024-02-01', '2024-03-01')" (after UNIONing views)
  - by_customer: "customer_id = 'C001'" or "customer_id IN (...)"
  - by_product: "product = 'Laptop'" or "product IN (...)"
  
- popular_query_templates:
  - "revenue by region": 
    dimensions: [region]
    filters: []
    grain: one row per region
    aggregation: sum(quantity * price) group by region
    query_strategy: Single month - use specific view. Multi-month - UNION all *_sales_* views first, then aggregate.
    
  - "top products": 
    dimensions: [product]
    filters: []
    grain: one row per product
    aggregation: sum(quantity * price) group by product order by sum(quantity * price) desc limit N
    query_strategy: Single month - use specific view. Multi-month - UNION all *_sales_* views first.
    
  - "sales by category": 
    dimensions: [category]
    filters: []
    grain: one row per category
    aggregation: sum(quantity * price) group by category
    query_strategy: Single month - use specific view. Multi-month - UNION all *_sales_* views first.
    
  - "revenue by region and category": 
    dimensions: [region, category]
    filters: []
    grain: one row per region-category combination
    aggregation: sum(quantity * price) group by region, category
    query_strategy: Single month - use specific view. Multi-month - UNION all *_sales_* views first.
    
  - "sales over time" or "trend analysis":
    dimensions: [report_date]
    filters: []
    grain: one row per report_date (month)
    aggregation: sum(quantity * price) group by report_date order by report_date
    query_strategy: **Must UNION all *_sales_* views first**, then group by report_date.
    example_sql:
      ```sql
      SELECT report_date, SUM(quantity * price) as revenue
      FROM (
        SELECT * FROM jan012024_sales_jan012024
        UNION ALL SELECT * FROM feb012024_sales_jan012024
        UNION ALL SELECT * FROM mar012024_sales_mar012024
      ) GROUP BY report_date ORDER BY report_date
      ```

---

## 7) Metric dictionary

- metrics:
  - metric_name: revenue
    definition: Sum of (quantity * price) from sales
    required_tables: ["*_sales_*"]
    union_strategy: "For multi-month queries, UNION ALL all views matching *_sales_* pattern. All views have same schema including report_date."
    example_sql_single_month: "SELECT SUM(quantity * price) FROM jan012024_sales_jan012024"
    example_sql_multi_month: |
      SELECT report_date, SUM(quantity * price) as revenue
      FROM (
        SELECT * FROM jan012024_sales_jan012024
        UNION ALL SELECT * FROM feb012024_sales_jan012024
        UNION ALL SELECT * FROM mar012024_sales_mar012024
      ) GROUP BY report_date

  - metric_name: order_count
    definition: Count of distinct order_id
    required_tables: ["*_sales_*"]
    union_strategy: "For multi-month queries, UNION ALL all views matching *_sales_* pattern first."

  - metric_name: avg_order_value
    definition: revenue / order_count
    required_tables: ["*_sales_*"]
    union_strategy: "Calculate revenue and order_count separately, then divide. For multi-month, UNION views first."

  - metric_name: stock_value
    definition: Sum of (stock_quantity * unit_cost)
    required_tables: ["*_inventory_*"]
    union_strategy: "For multi-month queries, UNION ALL all views matching *_inventory_* pattern first."

  - metric_name: total_purchases
    definition: Sum of total_purchases from customers table
    required_tables: ["*_customer_*"]
    union_strategy: "For multi-month queries, UNION ALL all views matching *_customer_* pattern first."

---

## 8) Join conventions

- canonical_joins:
  - left_table: *_sales_*
    right_table: *_customer_*
    on: {sales_view}.customer_id = {customer_view}.customer_id
    join_type: left
    join_strategy: "Join within same month: jan012024_sales_jan012024 JOIN jan012024_customer_jan012024. For multi-month queries, UNION sales views first, then join with appropriate customer view (or UNION customer views if needed)."
    example_single_month: |
      SELECT s.*, c.customer_name 
      FROM jan012024_sales_jan012024 s
      LEFT JOIN jan012024_customer_jan012024 c 
      ON s.customer_id = c.customer_id
    example_multi_month: |
      SELECT s.report_date, SUM(s.quantity * s.price) as revenue, c.customer_tier
      FROM (
        SELECT * FROM jan012024_sales_jan012024
        UNION ALL SELECT * FROM feb012024_sales_jan012024
        UNION ALL SELECT * FROM mar012024_sales_mar012024
      ) s
      LEFT JOIN jan012024_customer_jan012024 c ON s.customer_id = c.customer_id
      GROUP BY s.report_date, c.customer_tier

  - left_table: *_sales_*
    right_table: *_inventory_*
    on: {sales_view}.product = {inventory_view}.product_name
    join_type: left
    join_strategy: "Join within same month: jan012024_sales_jan012024 JOIN jan012024_inventory_jan012024. For multi-month, UNION sales views first, then join with appropriate inventory view."
    example_single_month: |
      SELECT s.*, i.unit_cost, i.stock_quantity
      FROM jan012024_sales_jan012024 s
      LEFT JOIN jan012024_inventory_jan012024 i 
      ON s.product = i.product_name

- forbidden_joins: []

---

## 9) Safety defaults (Executor reference)

- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false
  - union_limit: "When UNIONing multiple views, consider limiting number of views if query performance is an issue. Prefer single-month queries when possible for better performance."

---

## 10) Notes

- Time defaults are **optional**, not mandatory.
- If the user does not request time, use `no_time`.
- Investigation should always precede clarification.
- **View naming**: All views follow pattern `{date}_{entity}_{date}`. Use pattern matching (`*_sales_*`, `*_customer_*`, `*_inventory_*`) to identify relevant views.
- **Single vs Multi-month**: Prefer single-month queries when user specifies a specific month or when query doesn't require trend analysis. Use UNION ALL for multi-month queries, trend analysis, or "past N months" queries.
- **Date column**: The `report_date` column exists in all views and matches the folder date. Use this column for time-based filtering and grouping. No need to extract date from view names.
- **Schema consistency**: All views within the same entity type (sales, customer, inventory) have identical schemas, making UNION ALL straightforward.
- Default interpretation (PM policy): If user says "sales" without specifying, interpret as **revenue** = SUM(quantity * price) from the sales views (`*_sales_*`). For "in January", filter `report_date` to that month and prefer the matching monthly sales view (e.g., `jan012024_sales_jan012024`) when possible.

