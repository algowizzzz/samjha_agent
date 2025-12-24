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

## 4) Core entities (hints only)

- primary_entities:
  - name: customers
    typical_grain: one row per customer
    default_start_table_hint: sample_customer_data

  - name: products
    typical_grain: one row per product
    default_start_table_hint: sample_inventory_data

  - name: sales
    typical_grain: one row per order line
    default_start_table_hint: sample_sales_data

---

## 5) Common dimensions (Decider hint)

- common_dimensions:
  - region
  - product
  - customer_id
  - order_date

---

## 6) Metric dictionary

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

---

## 7) Join conventions

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

## 8) Safety defaults (Executor reference)

- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false

---

## 9) Notes

- Time defaults are **optional**, not mandatory.
- If the user does not request time, use `no_time`.
- Investigation should always precede clarification.
