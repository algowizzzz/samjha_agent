# Domain Configuration Fillable Template

Fill out this template to create your domain configuration file. Each section has tables you can fill in with your data.

> **📖 Need More Details?** See the full **[Creating Domain Configuration Guide](Creating_Domain_Configuration_Guide.md)** for explanations, examples, and best practices.

**Instructions**: 
1. Fill in each table below with your data
2. Review the examples in each section for format guidance
3. Copy the completed sections into markdown format (see template at bottom)
4. Save as: `domain_instructions/<your_domain>_domain.md`
5. Test with simple queries to verify configuration

---

## Section 1: Domain Identity

**What it is**: Basic identification of your domain - a name and description.

**Think of it as**: The "name tag" for your data warehouse.

**What you need**:
- **`domain_key`**: A short identifier (like a variable name) for your domain. Use lowercase, no spaces.
- **`description`**: A brief 1-3 sentence description of what data your domain covers.

**Ecommerce example**:
- `domain_key`: `ecomm`
- `description`: `ECommerce warehouse covering customers, inventory, and sales.`

**In your domain file**: This is the header that identifies your domain configuration.

---

| Field | Value Type | Example | Your Value | Notes |
|-------|-----------|---------|------------|-------|
| `domain_key` | lowercase string (3-20 chars) | `ecomm` | | Short identifier for your domain |
| `description` | 1-3 sentence string | `ECommerce warehouse covering customers, inventory, and sales.` | | Brief description of your data |

---

## Section 2: Time Semantics

**What it is**: Defines how time-based queries work in your domain.

**Think of it as**: Rules for when and how to filter data by time/date.

**What you need**:
- **`supports_no_time_queries`**: Can users ask queries without time constraints? 
  - `true` = Users can say "list all products" (no time filter)
  - `false` = All queries must have time context
- **`apply_default_time_rule_when`**: When should the system automatically apply time filters?
  - `explicit_or_implied_time_only`: Only when user mentions time or implies recency (e.g., "recent sales")
  - `always`: Always apply time defaults
  - `never`: Never apply automatically
- **`default_time_column`**: Your main date/time column (e.g., `order_date`, `transaction_date`)
- **`default_time_rule`**: How to filter by default
  - `last_n_days`: Last N days from today
  - `as_of_latest`: Latest available data point
  - `no_time`: No time filtering
- **`default_time_n_days`**: If using `last_n_days`, how many days? (e.g., 30 = last 30 days)

**Ecommerce example**:
- `supports_no_time_queries`: `true` (users can ask "list products" without time)
- `apply_default_time_rule_when`: `explicit_or_implied_time_only` (only apply when user mentions time)
- `default_time_column`: `order_date`
- `default_time_rule`: `last_n_days`
- `default_time_n_days`: `30`

**How it works**:
- User asks: "recent sales" → System applies `last_n_days: 30` filter automatically
- User asks: "list all products" → No time filter applied (because `supports_no_time_queries: true`)

**In your domain file**: These settings control when time filters are automatically applied to queries.

---

### Basic Time Settings

| Field | Value Type | Options | Example | Your Value | Notes |
|-------|-----------|---------|---------|------------|-------|
| `supports_no_time_queries` | boolean | `true` or `false` | `true` | | Allow queries without time constraints? |
| `apply_default_time_rule_when` | enum | `explicit_or_implied_time_only`<br>`always`<br>`never` | `explicit_or_implied_time_only` | | When to apply time defaults |
| `default_time_column` | string (column name) | Any date column | `order_date` | | Primary time column name |
| `default_time_rule` | enum | `last_n_days`<br>`as_of_latest`<br>`no_time` | `last_n_days` | | Default time filtering rule |
| `default_time_n_days` | integer | 7, 30, 90, 365 | `30` | | Number of days (if using last_n_days) |

### Time Columns by Entity

**What it is**: Maps each entity to its time column (if it has one).

**Think of it as**: Which date/time column to use for each entity when filtering by time.

**What you need**:
- For each entity from Section 4, specify its time column
- Some entities have time columns (e.g., `sales` has `order_date`)
- Some don't (e.g., `products` has no time - products don't have dates)
- Use `""` (empty string) if an entity has no time column

**Ecommerce example**:
- `sales` entity → `order_date` column (sales have dates)
- `customers` entity → `signup_date` column (customers have signup dates)
- `products` entity → `""` (products don't have dates)

**How it works**:
- When user asks "recent sales", system uses `order_date` from `sales` entity
- When user asks "new customers", system uses `signup_date` from `customers` entity
- When user asks "list products", no time column needed (products don't have dates)

**In your domain file**: This tells the system which time column to use for each entity.

---

Fill in one row per entity:

| Entity Name | Time Column | Your Value | Notes |
|-------------|-------------|------------|-------|
| | | | Entity name from Section 4 |
| | | | Column name or `""` if no time column |
| | | | |
| | | | |
| | | | |

**Example**:
| Entity Name | Time Column | Your Value | Notes |
|-------------|-------------|------------|-------|
| sales | order_date | | |
| customers | signup_date | | |
| products | "" | | No time column for products |

---

## Section 3: Listing Rules

**What it is**: Controls how "list" queries behave (e.g., "list customers", "list products").

**Think of it as**: Rules for simple listing queries that don't require metrics.

**What you need**:
- **`listing_allows_empty_metrics`**: Can users say "list customers" without specifying a metric? 
  - `true` = Yes, they can just list entities (e.g., "list all products")
  - `false` = They must specify a metric (e.g., "list customers with revenue")
- **`listing_default_limit`**: Maximum number of rows to return for listing queries. Prevents accidentally returning millions of rows.

**Ecommerce example**:
- `listing_allows_empty_metrics`: `true` (users can say "list products" without metrics)
- `listing_default_limit`: `50` (return max 50 rows)

**How it works**:
- User asks: "list all products" → System returns list of products (no metrics needed)
- User asks: "list top 10 products by revenue" → System returns products with revenue metric
- System limits results to 50 rows by default (prevents huge result sets)

**In your domain file**: These settings control whether users can make simple listing queries and how many rows to return.

---

| Field | Value Type | Options | Example | Your Value | Notes |
|-------|-----------|---------|---------|------------|-------|
| `listing_allows_empty_metrics` | boolean | `true` or `false` | `true` | | Allow "list X" queries without metrics? |
| `listing_default_limit` | integer | 10, 25, 50, 100, 200 | `50` | | Max rows for listing queries |

---

## Section 4: Core Entities

### 1. ENTITIES

**What it is**: Main business objects in your data.

**Think of it as**: The "things" your business tracks.

**Ecommerce example**:
- `customers` — people who buy
- `products` — items you sell
- `sales` — transactions/orders

**In your domain file**: List 3–5 main entities. These are the starting points for queries like "show me customers" or "list products".

**Note**: Entity names map to table names (e.g., entity `customers` → table `customer_data`)

---

### 2. GRAIN

**What it is**: What one row in a table represents.

**Think of it as**: The level of detail in your data.

**Ecommerce examples**:

| Table | Grain | What This Means |
|-------|-------|----------------|
| `customers` | one row per customer | Each row = one customer (C001, C002, C003...) |
| `products` | one row per product | Each row = one product (Laptop, Chair, Phone...) |
| `sales` | one row per order line | Each row = one item in an order (Order123 + Laptop, Order123 + Chair, Order124 + Phone...) |

**Why it matters**:
- If `sales` is "one row per order line", one order can have multiple rows (one per product).
- To get "revenue per customer", you group/aggregate the sales rows by customer.

**In your domain file**: For each entity, describe the grain so the system knows how to aggregate.

---

Fill in one row per entity (recommended: 3-5 entities):

| Entity Name | Typical Grain | Default Table Hint | Your Value | Notes |
|-------------|---------------|-------------------|------------|-------|
| | | | | Entity identifier (e.g., `customers`) |
| | | | | What one row represents (e.g., `one row per customer`) |
| | | | | Table name hint (e.g., `sample_customer_data`) |
| | | | | |
| | | | | |
| | | | | |

**Example**:
| Entity Name | Typical Grain | Default Table Hint | Your Value | Notes |
|-------------|---------------|-------------------|------------|-------|
| customers | one row per customer | sample_customer_data | | |
| products | one row per product | sample_inventory_data | | |
| sales | one row per order line | sample_sales_data | | |

### Grain Verification Steps (per entity)

**What it is**: Steps to verify the grain is correct when the system inspects your data.

**Think of it as**: Instructions for the system to check that it understands your data structure correctly.

**What you need**:
- For each entity, provide 2-3 verification steps
- Common steps: "Use inspect_table to verify grain", "Check for unique customer_id"
- These help the system confirm the grain when it looks at your actual data

**Ecommerce example**:
- Entity: `customers`
  - Step 1: "Use inspect_table to verify grain"
  - Step 2: "Check for unique customer_id"
  - Step 3: "Grain affects aggregation: customer-level metrics require grouping by customer_id"

**How it works**:
- System runs `inspect_table` on your data
- System checks for unique identifiers (e.g., customer_id)
- System confirms grain matches your specification

**In your domain file**: These steps help the system verify grain when investigating your data.

---

For each entity above, fill in verification steps:

| Entity | Verification Step 1 | Verification Step 2 | Verification Step 3 | Your Value |
|--------|-------------------|-------------------|-------------------|------------|
| | | | | Entity name |
| | | | | Step 1 (e.g., "Use inspect_table to verify grain") |
| | | | | Step 2 (e.g., "Check for unique customer_id") |
| | | | | Step 3 (optional) |

### Grain Examples (per entity)

**What it is**: Example queries that show how grain works for each entity.

**Think of it as**: Training examples showing the system how to interpret different query types.

**What you need**:
- For each entity, provide 2-3 example queries
- Format: `"user query" → grain: description`
- Show both simple listings and aggregations

**Ecommerce example**:
- Entity: `customers`
  - `"list customers" → grain: one row per customer` (simple listing from customer table)
  - `"customer purchases" → grain: one row per customer (aggregated from sales)` (needs aggregation from sales table)
- Entity: `sales`
  - `"top products" → grain: one row per product (aggregated from order lines)` (aggregate sales by product)
  - `"revenue by region" → grain: one row per region (aggregated from order lines)` (aggregate sales by region)

**How it works**:
- System sees user query: "customer purchases"
- System matches to example: "customer purchases" → grain: one row per customer (aggregated from sales)
- System knows to aggregate sales data grouped by customer

**In your domain file**: These examples help the system understand what grain to use for different query patterns.

---

For each entity, provide 2-3 query examples:

| Entity | Query Example 1 | Query Example 2 | Query Example 3 | Your Value |
|--------|----------------|----------------|----------------|------------|
| | | | | Entity name |
| | | | | Example query → grain description |
| | | | | Example query → grain description |
| | | | | Example query → grain description (optional) |

**Example**:
| Entity | Query Example 1 | Query Example 2 | Query Example 3 | Your Value |
|--------|----------------|----------------|----------------|------------|
| customers | "list customers" → grain: one row per customer | "customer purchases" → grain: one row per customer (aggregated from sales) | | |
| sales | "top products" → grain: one row per product (aggregated from order lines) | "revenue by region" → grain: one row per region (aggregated from order lines) | "sales by category" → grain: one row per category (aggregated from order lines) | |

---

## Section 5: Dimensions Dictionary

### 3. DIMENSIONS

**What it is**: Columns you use to group or break down data.

**Think of it as**: Ways to slice your data (by region, by product, by category, etc.).

**Ecommerce examples**:
- `region` — group sales by North/South/East/West
- `product` — group by product name
- `category` — group by Electronics/Furniture
- `customer_id` — group by customer
- `order_date` — group by time

**How it works**:
- User asks: "revenue by region"
- System uses the `region` dimension to group sales data
- Result: one row per region with total revenue

**In your domain file**: List columns users can group by, with the table and column name.

**Note**: Not all categorical columns are dimensions - only ones users will want to group by
- ✅ Dimensions: `region`, `category`, `product`, `customer_id`, `order_date`
- ❌ Not dimensions: `quantity`, `price` (these are metrics, not for grouping)

---

### 4. DIMENSIONS DICTIONARY

**What it is**: A catalog of all available dimensions with details.

**Think of it as**: A reference guide for each dimension.

**For each dimension, you provide**:
1. `dimension_name` — identifier (e.g., `region`)
2. `table` — which table has this column (e.g., `sample_sales_data`)
3. `column` — actual column name (e.g., `region`)
4. `description` — what it represents (e.g., "Sales region (North, South, East, West)")
5. `common_queries` — example queries using it (e.g., ["revenue by region", "sales by region"])
6. `synonyms` — alternative names users might say (e.g., `product` might have synonyms `["item", "sku"]`)
7. `usage_notes` — guidance on when/how to use it

**Ecommerce example**:
```yaml
- dimension_name: region
  table: sample_sales_data
  column: region
  description: Sales region (North, South, East, West)
  common_queries: ["revenue by region", "sales by region"]
  synonyms: []
  usage_notes: "Use for geographic breakdowns"
```

---

### How They Work Together

**Example query**: "Show me revenue by region and category"

1. **Entity**: Uses `sales` (transaction data)
2. **Grain**: `sales` is "one row per order line" — needs aggregation
3. **Dimensions**: Uses `region` and `category` to group
4. **Result**: One row per region-category combination with total revenue

**SQL equivalent**:
```sql
SELECT region, category, SUM(quantity * price) as revenue
FROM sales
GROUP BY region, category
```

---

Fill in one row per dimension (recommended: 5-10 dimensions):

| Dimension Name | Table | Column | Description | Your Value | Notes |
|----------------|-------|--------|-------------|------------|-------|
| | | | | | Dimension identifier |
| | | | | | Table where column exists |
| | | | | | Actual column name |
| | | | | | Human-readable description |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

**Example**:
| Dimension Name | Table | Column | Description | Your Value | Notes |
|----------------|-------|--------|-------------|------------|-------|
| region | sample_sales_data | region | Sales region (North, South, East, West) | | |
| product | sample_sales_data | product | Product name | | |
| category | sample_sales_data | category | Product category (e.g., Electronics, Furniture) | | |

### Dimension Details (per dimension)

**What it is**: Additional metadata for each dimension to help the system understand how to use it.

**Think of it as**: Extra information that helps the system recognize and use each dimension correctly.

**What you need**:
- **`common_queries`**: Example query patterns using this dimension (e.g., ["revenue by region", "sales by region"])
- **`synonyms`**: Alternative names users might say (e.g., `product` might have synonyms `["item", "sku"]`)
- **`usage_notes`**: Guidance on when/how to use this dimension

**Ecommerce example**:
- Dimension: `product`
  - `common_queries`: `["top products", "products by sales", "revenue by product"]`
  - `synonyms`: `["item", "sku"]` (users can say "revenue by item" or "revenue by sku")
  - `usage_notes`: `"Use for product-level analysis. Groups order lines by product name."`
- Dimension: `region`
  - `common_queries`: `["revenue by region", "sales by region", "total by region"]`
  - `synonyms`: `[]` (no synonyms)
  - `usage_notes`: `"Use for geographic breakdowns. Verify column exists via inspect_table."`

**How it works**:
- User says: "revenue by item" → System recognizes "item" is synonym for "product" → Uses `product` dimension
- User says: "sales by region" → System matches to common query pattern → Uses `region` dimension

**In your domain file**: These details help the system recognize user language and map it to dimensions.

---

For each dimension above, fill in additional details:

| Dimension Name | Common Queries (comma-separated) | Synonyms (comma-separated) | Usage Notes | Your Value |
|----------------|--------------------------------|---------------------------|-------------|------------|
| | | | | Dimension name |
| | | | | Query patterns (e.g., "revenue by region, sales by region") |
| | | | | Alternative names (e.g., "item, sku" or leave empty) |
| | | | | Guidance text |
| | | | | |
| | | | | |
| | | | | | |

**Example**:
| Dimension Name | Common Queries | Synonyms | Usage Notes | Your Value |
|----------------|----------------|----------|-------------|------------|
| region | revenue by region, sales by region, total by region | | Use for geographic breakdowns. Verify column exists via inspect_table. | |
| product | top products, products by sales, revenue by product | item, sku | Use for product-level analysis. Groups order lines by product name. | |

---

## Section 6: Common Filters

**What it is**: Default filters and filter patterns users commonly apply.

**Think of it as**: Pre-defined filters and filter templates that make querying easier.

**What you need**:
- **Default filters**: Filters automatically applied (e.g., "only active products", "only completed orders")
- **Filter patterns**: Reusable filter templates (e.g., "by region", "by date range")
- **Query templates**: Complete query patterns showing dimensions + filters + aggregation

**Ecommerce example**:
- Default filter: `active_products: "is_active = true"` - automatically filters to only active products
- Filter pattern: `by_region: "region IN ('North', 'South', 'East', 'West')"` - template for region filtering
- Query template: "revenue by region" with dimensions `[region]`, filters `[]`, aggregation `sum(quantity * price) group by region`

**How it works**:
- User asks: "revenue by region" → System uses query template → Applies dimensions and aggregation
- User asks: "active products" → System applies default filter `is_active = true` automatically
- User asks: "sales in North region" → System uses `by_region` pattern with value "North"

**In your domain file**: These help the system understand common filtering patterns and apply them automatically.

---

### Default Filters

**What it is**: Filters that should be automatically applied to queries.

**Think of it as**: Automatic data quality filters that ensure only valid records are shown.

**What you need**:
- SQL WHERE clause fragments that get added automatically
- Usually for data quality (e.g., only show active records, only completed transactions)
- Format: `filter_name: "SQL condition" (optional note)`

**Ecommerce example**:
- `active_products: "is_active = true" (if column exists in inventory table)` - only show active products
- `completed_orders: "order_status = 'completed'" (if column exists in sales table)` - only show completed orders

**How it works**:
- User asks: "list products" → System automatically adds `WHERE is_active = true` (if default filter exists)
- User asks: "revenue by category" → System automatically filters to completed orders only

**In your domain file**: These ensure data quality by automatically filtering out invalid or unwanted records.

---

Fill in default filters (optional, 0-3 filters):

| Filter Name | SQL Condition | Table/Column Note | Your Value | Notes |
|-------------|---------------|-------------------|------------|-------|
| | | | | Filter identifier |
| | | | | SQL WHERE clause fragment |
| | | | | Note about table/column (optional) |
| | | | | |
| | | | | |

**Example**:
| Filter Name | SQL Condition | Table/Column Note | Your Value | Notes |
|-------------|---------------|-------------------|------------|-------|
| active_products | is_active = true | if column exists in inventory table | | |
| completed_orders | order_status = 'completed' | if column exists in sales table | | |

### Filter Patterns

**What it is**: Reusable filter templates for common filtering scenarios.

**Think of it as**: Filter "recipes" the system can use when users request specific filters.

**What you need**:
- Filter pattern name (e.g., `by_region`, `by_category`)
- SQL pattern with examples (e.g., `"region IN ('North', 'South', 'East', 'West')" or "region = '...'"`)

**Ecommerce example**:
- `by_region: "region IN ('North', 'South', 'East', 'West')" or "region = '...'"`
- `by_category: "category = 'Electronics'" or "category IN (...)"`
- `by_date_range: "order_date BETWEEN '2024-01-01' AND '2024-12-31'"`

**How it works**:
- User asks: "sales in North region" → System uses `by_region` pattern → Applies `region = 'North'`
- User asks: "revenue by Electronics category" → System uses `by_category` pattern → Applies `category = 'Electronics'`
- User asks: "sales from January to March" → System uses `by_date_range` pattern → Applies date range filter

**In your domain file**: These patterns help the system understand how to construct filters from user requests.

---

Fill in common filter patterns (recommended: 3-5 patterns):

| Pattern Name | SQL Pattern | Example Values | Your Value | Notes |
|--------------|-------------|----------------|------------|-------|
| | | | | Pattern identifier |
| | | | | SQL pattern with examples |
| | | | | Example values (optional) |
| | | | | |
| | | | | |
| | | | | |

**Example**:
| Pattern Name | SQL Pattern | Example Values | Your Value | Notes |
|--------------|-------------|----------------|------------|-------|
| by_region | region IN ('North', 'South', 'East', 'West') or region = '...' | North, South, East, West | | |
| by_category | category = 'Electronics' or category IN (...) | Electronics, Furniture | | |
| by_date_range | order_date BETWEEN '2024-01-01' AND '2024-12-31' | | | |

### Popular Query Templates

**What it is**: Complete query patterns showing how dimensions, filters, and aggregation work together.

**Think of it as**: Full query "blueprints" that show the system how to build complex queries.

**What you need**:
- **Query pattern**: What the user asks (e.g., "revenue by region")
- **Dimensions**: What to GROUP BY (e.g., `[region]` or `[region, category]`)
- **Filters**: Any WHERE conditions (e.g., `[]` or `["region = 'North'"]`)
- **Grain**: Output structure (e.g., `one row per region`)
- **Aggregation**: SQL calculation (e.g., `sum(quantity * price) group by region`)

**Ecommerce example**:
- Query: "revenue by region"
  - Dimensions: `[region]`
  - Filters: `[]` (no filters)
  - Grain: `one row per region`
  - Aggregation: `sum(quantity * price) group by region`
- Query: "revenue by region and category"
  - Dimensions: `[region, category]`
  - Filters: `[]`
  - Grain: `one row per region-category combination`
  - Aggregation: `sum(quantity * price) group by region, category`

**How it works**:
- User asks: "revenue by region" → System matches to template → Uses dimensions `[region]`, applies aggregation
- System generates SQL: `SELECT region, SUM(quantity * price) as revenue FROM sales GROUP BY region`

**In your domain file**: These templates show the system how to construct complete queries from user requests.

---

Fill in query templates (recommended: 3-5 templates):

| Query Pattern | Dimensions (comma-separated) | Filters | Grain | Aggregation | Your Value |
|---------------|------------------------------|--------|-------|------------|------------|
| | | | | | Query description |
| | | | | | Dimension names |
| | | | | | Filter strings (or empty) |
| | | | | | Output grain description |
| | | | | | SQL aggregation pattern |
| | | | | | |
| | | | | | |
| | | | | | |

**Example**:
| Query Pattern | Dimensions | Filters | Grain | Aggregation | Your Value |
|---------------|-----------|--------|-------|------------|------------|
| revenue by region | region | | one row per region | sum(quantity * price) group by region | |
| top products | product | | one row per product | sum(quantity) group by product order by sum(quantity) desc limit N | |
| revenue by region and category | region, category | | one row per region-category combination | sum(quantity * price) group by region, category | |

---

## Section 7: Metric Dictionary

**What it is**: Defines how to calculate business metrics from your data.

**Think of it as**: A recipe book for calculating the numbers users care about.

**What you need**:
- **Metric name**: What users call it (e.g., `revenue`, `order_count`, `avg_order_value`)
- **Definition**: How to calculate it (SQL-like formula)
- **Required tables**: Which tables are needed to calculate this metric

**Common metric patterns**:
- **Sum**: `Sum of (column)` or `Sum of (column1 * column2)` 
  - Example: `revenue = Sum of (quantity * price)`
- **Count**: `Count of distinct id_column`
  - Example: `order_count = Count of distinct order_id`
- **Average**: `Average of (column)`
  - Example: `avg_price = Average of (price)`
- **Ratio**: `metric1 / metric2` (references other metrics)
  - Example: `avg_order_value = revenue / order_count`
- **Difference**: `Sum of (revenue) - Sum of (cost)`
  - Example: `profit = Sum of (revenue) - Sum of (cost)`

**Ecommerce example**:
- Metric: `revenue`
  - Definition: `Sum of (quantity * price)`
  - Required tables: `["sample_sales_data"]`
- Metric: `order_count`
  - Definition: `Count of distinct order_id`
  - Required tables: `["sample_sales_data"]`
- Metric: `avg_order_value`
  - Definition: `revenue / order_count` (references other metrics)
  - Required tables: `["sample_sales_data"]`

**How it works**:
- User asks: "show me revenue" → System uses `revenue` metric → Calculates `SUM(quantity * price)`
- User asks: "revenue by region" → System uses `revenue` metric + `region` dimension → Groups by region and sums

**In your domain file**: These definitions tell the system how to calculate each metric users might request.

---

Fill in one row per metric (recommended: 5-10 metrics):

| Metric Name | Definition (Calculation) | Required Tables (comma-separated) | Your Value | Notes |
|-------------|--------------------------|--------------------------------|------------|-------|
| | | | | Metric identifier |
| | | | | Calculation formula (SQL-like) |
| | | | | Table names needed |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

**Example**:
| Metric Name | Definition | Required Tables | Your Value | Notes |
|-------------|-----------|----------------|------------|-------|
| revenue | Sum of (quantity * price) | sample_sales_data | | |
| order_count | Count of distinct order_id | sample_sales_data | | |
| avg_order_value | revenue / order_count | sample_sales_data | | |
| stock_value | Sum of (stock_quantity * unit_cost) | sample_inventory_data | | |

**Common Metric Patterns**:
- Sum: `Sum of (column)` or `Sum of (column1 * column2)`
- Count: `Count of distinct id_column` or `Count of rows`
- Average: `Average of (column)` or `Sum of (column) / Count of (rows)`
- Ratio: `metric1 / metric2` (references other metrics)
- Difference: `Sum of (revenue) - Sum of (cost)`

---

## Section 8: Join Conventions

**What it is**: Defines how tables relate to each other (foreign key relationships).

**Think of it as**: A map showing how to connect your tables together.

**What you need**:
- **Left table**: Source table (e.g., `sales_data`)
- **Right table**: Target table to join (e.g., `customer_data`)
- **Join condition**: The ON clause showing how they connect (e.g., `sales_data.customer_id = customer_data.customer_id`)
- **Join type**: How to join:
  - `left` = All rows from left table (most common - preserves all sales even if customer missing)
  - `inner` = Only matching rows from both tables (only sales with valid customers)
  - `right` = All rows from right table (rarely used)
  - `full` = All rows from both tables (rarely used)

**Ecommerce example**:
- Join 1:
  - Left table: `sample_sales_data`
  - Right table: `sample_customer_data`
  - Join condition: `sample_sales_data.customer_id = sample_customer_data.customer_id`
  - Join type: `left` (get all sales, even if customer info missing)
- Join 2:
  - Left table: `sample_sales_data`
  - Right table: `sample_inventory_data`
  - Join condition: `sample_sales_data.product = sample_inventory_data.product_name`
  - Join type: `left` (get all sales, even if product info missing)

**How it works**:
- User asks: "revenue by customer name" → System needs customer names
- System uses join: `sales_data LEFT JOIN customer_data ON sales_data.customer_id = customer_data.customer_id`
- System can now group by customer name instead of just customer_id

**In your domain file**: These joins tell the system how to connect tables when queries need data from multiple tables.

---

Fill in one row per join (recommended: 3-5 joins):

| Left Table | Right Table | Join Condition (ON clause) | Join Type | Your Value | Notes |
|------------|-------------|---------------------------|-----------|------------|-------|
| | | | | | Source table name |
| | | | | | Target table name |
| | | | | | `table1.column = table2.column` |
| | | | | | `left`, `inner`, `right`, or `full` |
| | | | | | |
| | | | | | |
| | | | | | |

**Example**:
| Left Table | Right Table | Join Condition | Join Type | Your Value | Notes |
|------------|-------------|---------------|-----------|------------|-------|
| sample_sales_data | sample_customer_data | sample_sales_data.customer_id = sample_customer_data.customer_id | left | | |
| sample_sales_data | sample_inventory_data | sample_sales_data.product = sample_inventory_data.product_name | left | | |

**Join Type Guidelines**:
- `left`: All rows from left table (most common)
- `inner`: Only matching rows from both tables
- `right`: All rows from right table (rarely used)
- `full`: All rows from both tables (rarely used)

### Forbidden Joins

**What it is**: Joins that should never be used (for performance or logical reasons).

**Think of it as**: Join patterns that are dangerous or don't make sense.

**What you need**:
- List any join patterns to avoid
- Usually for performance (e.g., cross joins create huge result sets)
- Or logical reasons (e.g., joining unrelated tables)
- Leave empty `[]` if none

**Ecommerce example**:
- `forbidden_joins: []` (no forbidden joins in ecommerce example)

**Common forbidden patterns**:
- Cross joins (Cartesian products) - create millions of rows
- Joining unrelated tables (e.g., customers to products directly without sales)
- Joins that create data quality issues

**How it works**:
- System checks forbidden joins before generating SQL
- If a join matches a forbidden pattern, system blocks it or asks for clarification

**In your domain file**: These prevent the system from creating problematic joins that could cause performance issues or incorrect results.

---

List any joins to avoid (optional):

| Forbidden Join Pattern | Reason | Your Value | Notes |
|------------------------|--------|------------|-------|
| | | | Join pattern to avoid |
| | | | Reason (performance, logic, etc.) |
| | | | |
| | | | |

**Example**: Leave empty if none: `[]`

---

## Section 9: Safety Defaults

**What it is**: Performance and safety guardrails to prevent huge result sets or slow queries.

**Think of it as**: Safety limits that protect against accidentally creating massive queries.

**What you need**:
- **`default_limit`**: Maximum rows returned by default (prevents accidentally returning millions of rows)
  - Small datasets (< 10K rows): 100
  - Medium datasets (10K-1M rows): 50
  - Large datasets (> 1M rows): 25
- **`avoid_select_star`**: Should system avoid `SELECT *` queries?
  - `true` = Select specific columns (better performance, especially on large tables)
  - `false` = Allow `SELECT *` (can be slow on large tables)
- **`allow_cross_join`**: Allow cross joins (Cartesian products)?
  - `false` = Block cross joins (recommended - they create huge result sets)
  - `true` = Allow (rarely needed)

**Ecommerce example**:
- `default_limit`: `50` (medium dataset)
- `avoid_select_star`: `true` (select specific columns for performance)
- `allow_cross_join`: `false` (block dangerous cross joins)

**How it works**:
- User asks: "list all products" → System returns max 50 rows (even if there are 10,000 products)
- System generates: `SELECT product_id, product_name, price FROM products LIMIT 50` (not `SELECT *`)
- System blocks: Cross joins that would create millions of rows

**In your domain file**: These settings protect against performance issues and ensure queries complete quickly.

---

| Field | Value Type | Options | Example | Your Value | Notes |
|-------|-----------|---------|---------|------------|-------|
| `default_limit` | integer | 10, 25, 50, 100, 200, 500 | `50` | | Max rows returned by default |
| `avoid_select_star` | boolean | `true` or `false` | `true` | | Avoid SELECT * queries? |
| `allow_cross_join` | boolean | `true` or `false` | `false` | | Allow cross joins? (usually false) |

**Recommended Settings by Dataset Size**:
- Small (< 10K rows): limit=100, avoid_select_star=false, allow_cross_join=false
- Medium (10K-1M rows): limit=50, avoid_select_star=true, allow_cross_join=false
- Large (> 1M rows): limit=25, avoid_select_star=true, allow_cross_join=false

---

## Section 10: Notes

**What it is**: Important notes, caveats, or special instructions about your domain.

**Think of it as**: A place to document special rules, limitations, or important information about your data.

**What you need**:
- Free text section for anything important
- Common items to document:
  - **Time behavior**: "Time defaults are optional, not mandatory"
  - **Query rules**: "If user doesn't request time, use no_time"
  - **Data quality**: "Some columns may be NULL - handle with COALESCE"
  - **Business rules**: "Revenue calculations exclude refunded orders"
  - **Limitations**: "Historical data available from 2020-01-01 onwards"
  - **Special cases**: "Store IDs are unique across all locations"

**Ecommerce example**:
- "Time defaults are **optional**, not mandatory."
- "If the user does not request time, use `no_time`."
- "Investigation should always precede clarification."

**How it works**:
- System reads these notes when processing queries
- Notes help guide system behavior for edge cases
- Notes provide context for data quality or business rule issues

**In your domain file**: These notes help the system understand special circumstances or rules specific to your domain.

---

Fill in any important notes or caveats:

| Note Type | Content | Your Value | Notes |
|-----------|---------|------------|-------|
| | | | Time behavior, query rules, data quality, business rules, limitations, etc. |
| | | | |
| | | | |
| | | | |

**Example Notes**:
- Time defaults are **optional**, not mandatory.
- If the user does not request time, use `no_time`.
- Investigation should always precede clarification.

---

## Final Steps

1. **Review**: Check that all required fields are filled
2. **Validate**: Ensure table/column names match your actual data
3. **Format**: Copy sections into markdown format (see template below)
4. **Save**: Create file: `domain_instructions/<your_domain>_domain.md`
5. **Test**: Try simple queries to verify configuration

---

## Markdown Output Template

After filling the tables above, format your output like this:

```markdown
# Domain: <your_domain_key>

## 1) Domain identity
- domain_key: <your_domain_key>
- description: <your_description>

---

## 2) Time semantics (Decider reference)

- supports_no_time_queries: <true/false>
- apply_default_time_rule_when: <option>
- default_time_column: <column_name>
- default_time_rule: <option>
- default_time_n_days: <number>

### Time columns by entity
- time_columns_by_entity:
  - <entity1>: <column1>
  - <entity2>: <column2>
  - <entity3>: ""

---

## 3) Listing rules

- listing_allows_empty_metrics: <true/false>
- listing_default_limit: <number>

---

## 4) Core entities (hints only)

- primary_entities:
  - name: <entity1>
    typical_grain: <grain_description>
    default_start_table_hint: <table_name>
    grain_verification:
      - <verification_step_1>
      - <verification_step_2>
    grain_examples:
      - "<query_example_1>"
      - "<query_example_2>"

---

## 5) Dimensions dictionary (Decider reference)

- dimensions:
  - dimension_name: <dimension1>
    table: <table_name>
    column: <column_name>
    description: <description>
    common_queries: ["<query1>", "<query2>"]
    synonyms: ["<synonym1>", "<synonym2>"]
    usage_notes: "<notes>"

---

## 6) Common filters (Decider reference)

- default_filters:
  - <filter_name>: "<sql_condition>" (<note>)
  
- filter_patterns:
  - <pattern_name>: "<sql_pattern>"
  
- popular_query_templates:
  - "<query_pattern>": 
    dimensions: [<dimension1>, <dimension2>]
    filters: []
    grain: <grain_description>
    aggregation: <aggregation_pattern>

---

## 7) Metric dictionary

- metrics:
  - metric_name: <metric1>
    definition: <calculation>
    required_tables: ["<table1>"]

---

## 8) Join conventions

- canonical_joins:
  - left_table: <table1>
    right_table: <table2>
    on: <table1>.<column1> = <table2>.<column2>
    join_type: <left/inner/right/full>

- forbidden_joins: []

---

## 9) Safety defaults (Executor reference)

- performance_guardrails:
  - default_limit: <number>
  - avoid_select_star: <true/false>
  - allow_cross_join: <true/false>

---

## 10) Notes

- <your_note_1>
- <your_note_2>
```

---

## Quick Reference: Field Types

| Field Type | Format | Examples |
|------------|--------|----------|
| **boolean** | `true` or `false` | `true`, `false` |
| **string** | Text in quotes or unquoted | `"order_date"`, `ecomm` |
| **integer** | Number | `30`, `50`, `100` |
| **enum** | One of predefined options | `last_n_days`, `left`, `inner` |
| **array** | Comma-separated in brackets | `["region", "category"]`, `[]` |
| **object** | Key-value pairs | `{name: "value"}` |

---

**Need Help?** Refer to the full guide: `Creating_Domain_Configuration_Guide.md`

