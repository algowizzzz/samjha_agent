# Creating Your Domain Configuration Guide

This guide helps data owners create domain configuration files that enable natural language queries against their data warehouse.

> **💡 Quick Start**: Prefer filling out tables? Use the **[Fillable Template](Domain_Configuration_Fillable_Template.md)** instead - it provides structured tables you can fill in directly.

## Overview

A domain configuration file (`<your_domain>_domain.md`) tells the system:
- What entities exist in your data
- How to calculate metrics
- How tables relate to each other
- What dimensions and filters are available
- Time semantics and defaults

**File Location**: Place your domain file in `domain_instructions/<your_domain>_domain.md`

---

## Step-by-Step Guide

### Step 1: Domain Identity

**Purpose**: Basic identification of your domain.

```markdown
## 1) Domain identity
- domain_key: your_domain_name
- description: Brief description of your data warehouse
```

**Example**:
```markdown
- domain_key: ecomm
- description: ECommerce warehouse covering customers, inventory, and sales.
```

**Possible Values**:
- `domain_key`: Lowercase, alphanumeric, underscores allowed
  - Examples: `ecomm`, `retail`, `finance`, `healthcare`, `market_risk`, `supply_chain`
  - Format: Short identifier (3-20 characters)
- `description`: Free text, 1-3 sentences
  - Examples: 
    - `"ECommerce warehouse covering customers, inventory, and sales."`
    - `"Retail store chain data covering stores, products, sales transactions, and inventory."`
    - `"Financial trading data with positions, trades, and risk limits."`

**Ecommerce Example**:
```markdown
- domain_key: ecomm
- description: ECommerce warehouse covering customers, inventory, and sales.
```

**Your Turn**:
- Choose a short, lowercase domain key (e.g., `retail`, `finance`, `healthcare`)
- Write 1-2 sentences describing what data you have

---

### Step 2: Time Semantics

**Purpose**: Define how time-based queries work in your domain.

```markdown
## 2) Time semantics (Decider reference)

- supports_no_time_queries: true/false
- apply_default_time_rule_when: explicit_or_implied_time_only / always / never
- default_time_column: column_name
- default_time_rule: last_n_days / as_of_latest / no_time
- default_time_n_days: 30

### Time columns by entity
- time_columns_by_entity:
  - entity_name: column_name
  - entity_name: ""
```

**Key Questions**:
1. **Do you have time-based data?** (e.g., transaction dates, order dates, event timestamps)
   - If yes: set `supports_no_time_queries: true`
   - If no: set `supports_no_time_queries: false` and `default_time_rule: no_time`

2. **What is your primary time column?** (e.g., `order_date`, `transaction_date`, `event_timestamp`)
   - This becomes `default_time_column`

3. **What time range makes sense by default?** (e.g., last 30 days, last 7 days)
   - Set `default_time_n_days` accordingly

**Example**:
```markdown
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
```

**Possible Values**:
- `supports_no_time_queries`: `true` or `false`
  - `true`: Users can ask queries without time constraints (e.g., "list all products")
  - `false`: All queries must have time context
- `apply_default_time_rule_when`: One of:
  - `explicit_or_implied_time_only`: Apply defaults only when user mentions time or implies recency
  - `always`: Always apply time defaults unless user explicitly says "all time"
  - `never`: Never apply time defaults automatically
- `default_time_column`: Column name (string) or empty string
  - Examples: `order_date`, `transaction_date`, `event_timestamp`, `created_at`, `sale_date`
  - Must match actual column name in your data
- `default_time_rule`: One of:
  - `last_n_days`: Last N days from current date
  - `as_of_latest`: Latest available data point
  - `no_time`: No time filtering
- `default_time_n_days`: Integer (typically 7, 30, 90, 365)
  - Only used when `default_time_rule: last_n_days`
- `time_columns_by_entity`: List of entity-to-column mappings
  - Format: `entity_name: column_name` or `entity_name: ""` (empty string if no time column)
  - Entity names should match your `primary_entities` names

**Ecommerce Example**:
```markdown
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
```

**Your Turn**:
- Identify your main time column
- Decide on default time window
- Map time columns for each entity (use `""` if entity has no time column)

---

### Step 3: Listing Rules

**Purpose**: Control how "list" queries behave (e.g., "list customers", "list products").

```markdown
## 3) Listing rules

- listing_allows_empty_metrics: true/false
- listing_default_limit: 50
```

**Key Questions**:
1. **Can users list entities without metrics?** (e.g., "list all customers")
   - If yes: `listing_allows_empty_metrics: true`
   - If no: `listing_allows_empty_metrics: false`

2. **What's a reasonable default limit?** (prevents huge result sets)
   - Common: 50, 100, or 200

**Example**:
```markdown
- listing_allows_empty_metrics: true
- listing_default_limit: 50
```

**Possible Values**:
- `listing_allows_empty_metrics`: `true` or `false`
  - `true`: Users can query "list customers" without specifying a metric
  - `false`: All queries must include at least one metric
- `listing_default_limit`: Integer (typically 10, 25, 50, 100, 200)
  - Maximum number of rows returned for listing queries
  - Prevents accidentally returning millions of rows

**Ecommerce Example**:
```markdown
- listing_allows_empty_metrics: true
- listing_default_limit: 50
```

**Your Turn**:
- Set whether listing without metrics is allowed
- Choose a safe default limit

---

### Step 4: Core Entities

**Purpose**: Define the main entities in your data and their grain (granularity).

```markdown
## 4) Core entities (hints only)

- primary_entities:
  - name: entity_name
    typical_grain: one row per [what]
    default_start_table_hint: table_name
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique [id_column]
    grain_examples:
      - "list [entity]" → grain: one row per [what]
      - "[entity] [metric]" → grain: one row per [what] (aggregated from [source])
```

**Key Questions**:
1. **What are your main entities?** (e.g., customers, products, orders, transactions)
2. **What is the grain of each entity?** (one row per customer, one row per order line, etc.)
3. **What table contains this entity?** (actual table name in your database)

**Understanding Grain**:
- **Grain** = what one row represents
- Examples:
  - `customers`: one row per customer
  - `sales`: one row per order line (multiple rows per order)
  - `orders`: one row per order

**Example**:
```markdown
- primary_entities:
  - name: customers
    typical_grain: one row per customer
    default_start_table_hint: sample_customer_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique customer_id
    grain_examples:
      - "list customers" → grain: one row per customer
      - "customer purchases" → grain: one row per customer (aggregated from sales)

  - name: sales
    typical_grain: one row per order line
    default_start_table_hint: sample_sales_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique order_id + product combinations
    grain_examples:
      - "top products" → grain: one row per product (aggregated from order lines)
```

**Possible Values**:
- `name`: Entity identifier (string)
  - Examples: `customers`, `products`, `sales`, `orders`, `transactions`, `inventory`
  - Should be plural noun matching your business terminology
- `typical_grain`: Description of what one row represents
  - Format: `"one row per [what]"`
  - Examples:
    - `"one row per customer"`
    - `"one row per order line"`
    - `"one row per transaction"`
    - `"one row per product"`
    - `"one row per store"`
- `default_start_table_hint`: Table name hint (string)
  - Examples: `sample_customer_data`, `sales_data`, `product_inventory`
  - Should match actual table/file name (without path or extension)
- `grain_verification`: List of verification steps (array of strings)
  - Examples:
    - `["Use inspect_table to verify grain", "Check for unique customer_id"]`
    - `["Use inspect_table to verify grain", "Check for unique order_id + product combinations"]`
- `grain_examples`: List of query-to-grain mappings (array of strings)
  - Format: `"query pattern" → grain: description`
  - Examples:
    - `"list customers" → grain: one row per customer`
    - `"customer purchases" → grain: one row per customer (aggregated from sales)`
    - `"top products" → grain: one row per product (aggregated from order lines)`

**Ecommerce Example**:
```markdown
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
```

**Your Turn**:
- List 3-5 main entities in your data
- For each entity:
  - Describe the grain
  - Provide the table name
  - Add verification steps
  - Give 2-3 example queries

---

### Step 5: Dimensions Dictionary

**Purpose**: Define dimensions (grouping/breakdown columns) available for analysis.

```markdown
## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: dimension_key
    table: table_name
    column: column_name
    description: What this dimension represents
    common_queries: ["query pattern 1", "query pattern 2"]
    synonyms: ["alternative name 1", "alternative name 2"]
    usage_notes: "When and how to use this dimension"
```

**Key Questions**:
1. **What columns can users group by?** (e.g., region, category, product, customer segment)
2. **What are common ways users refer to these?** (synonyms)
3. **What queries use this dimension?**

**Common Dimensions**:
- Geographic: region, country, state, city
- Product: category, brand, product_name
- Customer: customer_segment, customer_type
- Time: order_date, transaction_date (if not in time semantics)
- Organizational: department, division, team

**Example**:
```markdown
- dimensions:
  - dimension_name: region
    table: sample_sales_data
    column: region
    description: Sales region (North, South, East, West)
    common_queries: ["revenue by region", "sales by region", "total by region"]
    synonyms: []
    usage_notes: "Use for geographic breakdowns. Verify column exists via inspect_table before SQL generation."

  - dimension_name: category
    table: sample_sales_data
    column: category
    description: Product category (e.g., Electronics, Furniture)
    common_queries: ["revenue by category", "sales by category", "products by category"]
    synonyms: []
    usage_notes: "Use for category-level aggregation. Verify column exists via inspect_table."
```

**Possible Values**:
- `dimension_name`: Dimension identifier (string)
  - Examples: `region`, `category`, `product`, `customer_id`, `order_date`, `store_id`
  - Used as key for referencing in queries
- `table`: Table name where dimension column exists (string)
  - Examples: `sample_sales_data`, `customer_data`, `product_inventory`
  - Must match actual table/file name
- `column`: Actual column name in the table (string)
  - Examples: `region`, `category`, `product`, `customer_id`, `order_date`
  - Must match exact column name (case-sensitive)
- `description`: Human-readable description (string)
  - Examples:
    - `"Sales region (North, South, East, West)"`
    - `"Product category (e.g., Electronics, Furniture)"`
    - `"Customer identifier"`
  - Include example values if helpful
- `common_queries`: Array of query patterns (list of strings)
  - Examples:
    - `["revenue by region", "sales by region", "total by region"]`
    - `["top products", "products by sales", "revenue by product"]`
    - `["customer purchases", "sales by customer", "top customers"]`
- `synonyms`: Alternative names users might use (array of strings)
  - Examples:
    - `["item", "sku"]` for product dimension
    - `["location", "branch"]` for store dimension
    - `[]` if no synonyms
- `usage_notes`: Guidance on when/how to use (string)
  - Examples:
    - `"Use for geographic breakdowns. Verify column exists via inspect_table before SQL generation."`
    - `"Use for product-level analysis. Groups order lines by product name."`
    - `"May require join to customer_data for customer names."`

**Ecommerce Example**:
```markdown
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
```

**Your Turn**:
- List 5-10 key dimensions in your data
- For each dimension:
  - Specify table and column
  - Write a clear description
  - List 2-3 common query patterns
  - Add synonyms if applicable
  - Include usage notes

---

### Step 6: Common Filters

**Purpose**: Define default filters and filter patterns users might apply.

```markdown
## 6) Common filters (Decider reference)

- default_filters:
  - filter_name: "column = value" (if column exists in table)
  
- filter_patterns:
  - by_dimension: "dimension_column = 'value'" or "dimension_column IN (...)"
  - by_date_range: "date_column BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'"
  
- popular_query_templates:
  - "query pattern": 
    dimensions: [dimension1, dimension2]
    filters: []
    grain: one row per [what]
    aggregation: sum(metric) group by [dimensions]
```

**Key Questions**:
1. **What default filters make sense?** (e.g., active products only, completed orders only)
2. **What filter patterns are common?** (by region, by date range, by status)
3. **What are your most popular query patterns?**

**Example**:
```markdown
- default_filters:
  - active_products: "is_active = true" (if column exists in inventory table)
  - completed_orders: "order_status = 'completed'" (if column exists in sales table)
  
- filter_patterns:
  - by_region: "region IN ('North', 'South', 'East', 'West')" or "region = '...'"
  - by_category: "category = 'Electronics'" or "category IN (...)"
  - by_date_range: "order_date BETWEEN '2024-01-01' AND '2024-12-31'"
  
- popular_query_templates:
  - "revenue by region": 
    dimensions: [region]
    filters: []
    grain: one row per region
    aggregation: sum(quantity * price) group by region
```

**Possible Values**:
- `default_filters`: List of default filter definitions
  - Format: `filter_name: "SQL condition" (optional note)`
  - Examples:
    - `active_products: "is_active = true" (if column exists in inventory table)`
    - `completed_orders: "order_status = 'completed'" (if column exists in sales table)`
    - `active_customers: "status = 'active'" (if column exists in customer table)`
  - SQL conditions should be valid WHERE clause fragments
- `filter_patterns`: List of reusable filter patterns
  - Format: `pattern_name: "SQL pattern with examples"`
  - Examples:
    - `by_region: "region IN ('North', 'South', 'East', 'West')" or "region = '...'"`
    - `by_category: "category = 'Electronics'" or "category IN (...)"`
    - `by_date_range: "order_date BETWEEN '2024-01-01' AND '2024-12-31'"`
    - `by_customer: "customer_id = 'C001'" or "customer_id IN (...)"`
    - `by_product: "product = 'Laptop'" or "product IN (...)"`
- `popular_query_templates`: List of query pattern templates
  - Each template has:
    - `"query pattern"`: String describing the query (e.g., `"revenue by region"`)
    - `dimensions`: Array of dimension names (e.g., `[region]` or `[region, category]`)
    - `filters`: Array of filter strings (e.g., `[]` or `["region = 'North'"]`)
    - `grain`: String describing output grain (e.g., `"one row per region"`)
    - `aggregation`: String describing SQL aggregation (e.g., `"sum(quantity * price) group by region"`)

**Ecommerce Example**:
```markdown
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
```

**Your Turn**:
- List 2-3 default filters (if applicable)
- Define 3-5 common filter patterns
- Document 3-5 popular query templates with their structure

---

### Step 7: Metric Dictionary

**Purpose**: Define how to calculate business metrics from your data.

```markdown
## 7) Metric dictionary

- metrics:
  - metric_name: metric_key
    definition: How to calculate this metric
    required_tables: ["table1", "table2"]
```

**Key Questions**:
1. **What metrics do users care about?** (e.g., revenue, profit, count of orders, average order value)
2. **How is each metric calculated?** (SQL formula or description)
3. **Which tables are needed?**

**Common Metrics**:
- Revenue: `sum(quantity * price)`
- Count: `count(distinct id)`
- Average: `avg(column)`
- Ratio: `metric1 / metric2`

**Example**:
```markdown
- metrics:
  - metric_name: revenue
    definition: Sum of (quantity * price)
    required_tables: ["sample_sales_data"]

  - metric_name: order_count
    definition: Count of distinct order_id
    required_tables: ["sample_sales_data"]

  - metric_name: avg_order_value
    definition: revenue / order_count
    required_tables: ["sample_sales_data"]

  - metric_name: stock_value
    definition: Sum of (stock_quantity * unit_cost)
    required_tables: ["sample_inventory_data"]
```

**Possible Values**:
- `metric_name`: Metric identifier (string)
  - Examples: `revenue`, `order_count`, `avg_order_value`, `profit`, `stock_value`
  - Should be clear, business-friendly name
- `definition`: Calculation formula (string)
  - Format: SQL-like expression or natural language description
  - Examples:
    - `"Sum of (quantity * price)"`
    - `"Count of distinct order_id"`
    - `"revenue / order_count"` (for derived metrics)
    - `"Sum of (stock_quantity * unit_cost)"`
    - `"Average of (price)"`
    - `"Sum of (revenue) - Sum of (cost)"`
  - Can reference other metrics (e.g., `revenue / order_count`)
- `required_tables`: Array of table names (list of strings)
  - Examples: `["sample_sales_data"]`, `["sample_sales_data", "sample_customer_data"]`
  - Must include all tables needed to calculate the metric
  - Use actual table/file names

**Ecommerce Example**:
```markdown
- metrics:
  - metric_name: revenue
    definition: Sum of (quantity * price)
    required_tables: ["sample_sales_data"]

  - metric_name: order_count
    definition: Count of distinct order_id
    required_tables: ["sample_sales_data"]

  - metric_name: avg_order_value
    definition: revenue / order_count
    required_tables: ["sample_sales_data"]

  - metric_name: stock_value
    definition: Sum of (stock_quantity * unit_cost)
    required_tables: ["sample_inventory_data"]
```

**Common Metric Patterns**:
- **Sum metrics**: `"Sum of (column)"` or `"Sum of (column1 * column2)"`
- **Count metrics**: `"Count of distinct id_column"` or `"Count of rows"`
- **Average metrics**: `"Average of (column)"` or `"Sum of (column) / Count of (rows)"`
- **Ratio metrics**: `"metric1 / metric2"` (references other metrics)
- **Difference metrics**: `"Sum of (revenue) - Sum of (cost)"`

**Your Turn**:
- List 5-10 key metrics in your domain
- For each metric:
  - Provide clear calculation formula
  - List required tables
  - Use SQL-like syntax when possible

---

### Step 8: Join Conventions

**Purpose**: Define how tables relate to each other.

```markdown
## 8) Join conventions

- canonical_joins:
  - left_table: table1
    right_table: table2
    on: table1.foreign_key = table2.primary_key
    join_type: left / inner / right

- forbidden_joins: []
```

**Key Questions**:
1. **How do your main tables join?** (foreign key relationships)
2. **What join type is appropriate?** (left, inner, right)
3. **Are there any joins to avoid?** (performance or logical reasons)

**Example**:
```markdown
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
```

**Possible Values**:
- `canonical_joins`: List of join definitions
  - Each join has:
    - `left_table`: Source table name (string)
      - Examples: `sample_sales_data`, `customer_data`, `product_inventory`
    - `right_table`: Target table name (string)
      - Examples: `sample_customer_data`, `sample_inventory_data`
    - `on`: Join condition (string, SQL ON clause)
      - Format: `table1.column = table2.column`
      - Examples:
        - `sample_sales_data.customer_id = sample_customer_data.customer_id`
        - `sample_sales_data.product = sample_inventory_data.product_name`
      - Must use full table.column notation
    - `join_type`: Type of join (string)
      - Options: `left`, `inner`, `right`, `full`
      - Most common: `left` (preserves all rows from left table)
- `forbidden_joins`: List of join patterns to avoid (array of strings)
  - Examples: `[]` (empty if none)
  - Use if certain joins are:
    - Performance issues (e.g., cross joins)
    - Logically incorrect
    - Not supported by your data model

**Ecommerce Example**:
```markdown
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
```

**Join Type Guidelines**:
- `left`: Use when you want all rows from left table, even if no match in right table
  - Example: All sales, even if customer info missing
- `inner`: Use when you only want rows with matches in both tables
  - Example: Only sales with valid customer records
- `right`: Rarely used; preserves all rows from right table
- `full`: Rarely used; preserves all rows from both tables

**Your Turn**:
- Document 3-5 key join relationships
- Specify join conditions (on clause)
- Choose appropriate join types
- List any forbidden joins (if any)

---

### Step 9: Safety Defaults

**Purpose**: Set performance and safety guardrails.

```markdown
## 9) Safety defaults (Executor reference)

- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false
```

**Key Questions**:
1. **What's a safe default row limit?** (prevents huge result sets)
2. **Should SELECT * be avoided?** (usually yes for performance)
3. **Are cross joins allowed?** (usually no, very expensive)

**Example**:
```markdown
- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false
```

**Possible Values**:
- `performance_guardrails`: Object with safety settings
  - `default_limit`: Integer (typically 10, 25, 50, 100, 200, 500)
    - Maximum rows returned by default
    - Prevents accidentally returning millions of rows
    - Users can override with explicit limits in queries
  - `avoid_select_star`: `true` or `false`
    - `true`: System will select specific columns instead of `SELECT *`
    - `false`: Allows `SELECT *` (not recommended for large tables)
    - Recommended: `true` for performance
  - `allow_cross_join`: `true` or `false`
    - `true`: Allows cross joins (Cartesian products)
    - `false`: Blocks cross joins (recommended)
    - Cross joins can create huge result sets (millions of rows)
    - Recommended: `false` unless you have specific use cases

**Ecommerce Example**:
```markdown
- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false
```

**Recommended Settings**:
- Small datasets (< 10K rows): `default_limit: 100`, `avoid_select_star: false`, `allow_cross_join: false`
- Medium datasets (10K-1M rows): `default_limit: 50`, `avoid_select_star: true`, `allow_cross_join: false`
- Large datasets (> 1M rows): `default_limit: 25`, `avoid_select_star: true`, `allow_cross_join: false`

**Your Turn**:
- Set a reasonable default limit
- Disable SELECT * (recommended)
- Disable cross joins (recommended)

---

### Step 10: Notes

**Purpose**: Add any important notes or caveats.

```markdown
## 10) Notes

- Time defaults are **optional**, not mandatory.
- If the user does not request time, use `no_time`.
- Investigation should always precede clarification.
```

**Possible Values**:
- Free text section for important notes
- Common items to document:
  - Time behavior: `"Time defaults are **optional**, not mandatory."`
  - Query behavior: `"If the user does not request time, use `no_time`."`
  - Investigation rules: `"Investigation should always precede clarification."`
  - Data quality: `"Some columns may be NULL. Handle with COALESCE in metrics."`
  - Business rules: `"Revenue calculations exclude refunded orders."`
  - Limitations: `"Historical data available from 2020-01-01 onwards."`
  - Special cases: `"Store IDs are unique across all locations."`

**Ecommerce Example**:
```markdown
## 10) Notes

- Time defaults are **optional**, not mandatory.
- If the user does not request time, use `no_time`.
- Investigation should always precede clarification.
```

**Your Turn**:
- Add any domain-specific notes
- Document assumptions or limitations
- Include important usage guidelines

---

## Complete Template

Here's a complete template you can copy and fill in:

```markdown
# Domain: your_domain_name

## 1) Domain identity
- domain_key: your_domain_name
- description: Brief description of your data warehouse

---

## 2) Time semantics (Decider reference)

- supports_no_time_queries: true
- apply_default_time_rule_when: explicit_or_implied_time_only
- default_time_column: your_time_column
- default_time_rule: last_n_days
- default_time_n_days: 30

### Time columns by entity
- time_columns_by_entity:
  - entity1: time_column1
  - entity2: time_column2
  - entity3: ""

---

## 3) Listing rules

- listing_allows_empty_metrics: true
- listing_default_limit: 50

---

## 4) Core entities (hints only)

- primary_entities:
  - name: entity1
    typical_grain: one row per [what]
    default_start_table_hint: table_name
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique [id_column]
    grain_examples:
      - "list [entity]" → grain: one row per [what]
      - "[entity] [metric]" → grain: one row per [what] (aggregated from [source])

---

## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: dimension1
    table: table_name
    column: column_name
    description: What this dimension represents
    common_queries: ["query pattern 1", "query pattern 2"]
    synonyms: []
    usage_notes: "When and how to use this dimension"

---

## 6) Common filters (Decider reference)

- default_filters:
  - filter_name: "column = value" (if column exists in table)
  
- filter_patterns:
  - by_dimension: "dimension_column = 'value'" or "dimension_column IN (...)"
  
- popular_query_templates:
  - "query pattern": 
    dimensions: [dimension1]
    filters: []
    grain: one row per [what]
    aggregation: sum(metric) group by [dimensions]

---

## 7) Metric dictionary

- metrics:
  - metric_name: metric1
    definition: How to calculate this metric
    required_tables: ["table1"]

---

## 8) Join conventions

- canonical_joins:
  - left_table: table1
    right_table: table2
    on: table1.foreign_key = table2.primary_key
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

- Add your domain-specific notes here
```

---

## Best Practices

1. **Start Simple**: Begin with 3-5 entities, 5-10 dimensions, and 5-10 metrics. You can expand later.

2. **Be Specific**: Use actual table and column names from your database.

3. **Document Grain Clearly**: Understanding grain is critical for correct aggregations.

4. **Test Incrementally**: After creating your domain file, test with simple queries first.

5. **Iterate**: Domain files are living documents. Update as you discover new patterns.

6. **Use Examples**: Include query examples in grain_examples and common_queries to help the system understand intent.

7. **Verify Assumptions**: Use `inspect_table` tool to verify column names and grain before finalizing.

---

## Example: Retail Domain

Here's a complete example for a retail domain:

```markdown
# Domain: retail

## 1) Domain identity
- domain_key: retail
- description: Retail store chain data covering stores, products, sales transactions, and inventory.

---

## 2) Time semantics (Decider reference)

- supports_no_time_queries: true
- apply_default_time_rule_when: explicit_or_implied_time_only
- default_time_column: sale_date
- default_time_rule: last_n_days
- default_time_n_days: 30

### Time columns by entity
- time_columns_by_entity:
  - sales: sale_date
  - stores: opening_date
  - products: ""

---

## 3) Listing rules

- listing_allows_empty_metrics: true
- listing_default_limit: 50

---

## 4) Core entities (hints only)

- primary_entities:
  - name: stores
    typical_grain: one row per store
    default_start_table_hint: store_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique store_id
    grain_examples:
      - "list stores" → grain: one row per store
      - "store performance" → grain: one row per store (aggregated from sales)

  - name: products
    typical_grain: one row per product
    default_start_table_hint: product_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique product_id
    grain_examples:
      - "list products" → grain: one row per product
      - "top products" → grain: one row per product (aggregated from sales)

  - name: sales
    typical_grain: one row per transaction line
    default_start_table_hint: sales_data
    grain_verification:
      - Use inspect_table to verify grain
      - Check for unique transaction_id + product_id combinations
    grain_examples:
      - "sales by store" → grain: one row per store (aggregated from transaction lines)
      - "sales by product" → grain: one row per product (aggregated from transaction lines)

---

## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: store
    table: sales_data
    column: store_id
    description: Store identifier
    common_queries: ["sales by store", "revenue by store", "performance by store"]
    synonyms: ["location", "branch"]
    usage_notes: "Use for store-level analysis. May require join to store_data for store names."

  - dimension_name: product_category
    table: sales_data
    column: category
    description: Product category (e.g., Electronics, Clothing, Food)
    common_queries: ["sales by category", "revenue by category", "products by category"]
    synonyms: []
    usage_notes: "Use for category-level aggregation."

  - dimension_name: sale_date
    table: sales_data
    column: sale_date
    description: Date of sale transaction
    common_queries: ["sales over time", "revenue by date", "daily sales"]
    synonyms: []
    usage_notes: "Use for time-based analysis. Can be grouped by day/month/year."

---

## 6) Common filters (Decider reference)

- default_filters:
  - active_stores: "is_active = true" (if column exists in store_data)
  - completed_sales: "status = 'completed'" (if column exists in sales_data)
  
- filter_patterns:
  - by_store: "store_id = 'S001'" or "store_id IN (...)"
  - by_category: "category = 'Electronics'" or "category IN (...)"
  - by_date_range: "sale_date BETWEEN '2024-01-01' AND '2024-12-31'"
  
- popular_query_templates:
  - "sales by store": 
    dimensions: [store_id]
    filters: []
    grain: one row per store
    aggregation: sum(quantity * price) group by store_id
    
  - "top products": 
    dimensions: [product_id]
    filters: []
    grain: one row per product
    aggregation: sum(quantity) group by product_id order by sum(quantity) desc limit N

---

## 7) Metric dictionary

- metrics:
  - metric_name: revenue
    definition: Sum of (quantity * price)
    required_tables: ["sales_data"]

  - metric_name: transaction_count
    definition: Count of distinct transaction_id
    required_tables: ["sales_data"]

  - metric_name: avg_transaction_value
    definition: revenue / transaction_count
    required_tables: ["sales_data"]

  - metric_name: inventory_value
    definition: Sum of (stock_quantity * unit_cost)
    required_tables: ["inventory_data"]

---

## 8) Join conventions

- canonical_joins:
  - left_table: sales_data
    right_table: store_data
    on: sales_data.store_id = store_data.store_id
    join_type: left

  - left_table: sales_data
    right_table: product_data
    on: sales_data.product_id = product_data.product_id
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
- Store IDs are unique across all locations.
```

---

## Next Steps

1. **Create your domain file** using the template above
2. **Save it** to `domain_instructions/<your_domain>_domain.md`
3. **Test with simple queries** to verify the configuration
4. **Iterate** based on query patterns you discover
5. **Update** as your data model evolves

---

## Getting Help

If you encounter issues:
1. Review the example `ecomm_domain.md` file
2. Check that table and column names match your actual database
3. Verify grain definitions are accurate
4. Test with `inspect_table` tool to confirm schema

---

## Summary Checklist

Before finalizing your domain file, ensure you have:

- [ ] Domain identity (key and description)
- [ ] Time semantics configured
- [ ] Listing rules set
- [ ] 3-5 core entities with grain definitions
- [ ] 5-10 dimensions documented
- [ ] Common filters and query templates
- [ ] 5-10 metrics with clear definitions
- [ ] Join conventions documented
- [ ] Safety defaults configured
- [ ] Notes section completed

Good luck creating your domain configuration!

