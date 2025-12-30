# Test Queries for ecommerce_advanced Agent

## Data Structure Summary
- **Entities**: sales, inventory, customers
- **Time periods**: Jan 2024, Feb 2024, Mar 2024 (3 months)
- **Dimensions**: region, product, category, customer_id, report_date, supplier, customer_tier
- **Metrics**: revenue, order_count, avg_order_value, stock_value, total_purchases
- **Joins**: sales↔customer (customer_id), sales↔inventory (product=product_name)

---

## 10 Test Queries

### 1. `top products by revenue`
**Type**: Single entity, aggregation  
**Tests**: metric.definition usage, dimension inference (product), single vs multi-month handling  
**Expected**: UNION ALL all *_sales_* views, GROUP BY product, SUM(quantity * price), ORDER BY DESC, LIMIT

### 2. `revenue by region and category for January`
**Type**: Single entity, multi-dimension, single month  
**Tests**: date filtering, multiple dimensions, grain inference (region-category combo)  
**Expected**: Single view (jan012024_sales_jan012024), GROUP BY region, category, WHERE report_date = '2024-01-01'

### 3. `inventory stock value month over month`
**Type**: Multi-month trend, different entity  
**Tests**: UNION ALL strategy, aggregation_plan, inventory entity handling  
**Expected**: UNION ALL *_inventory_* views, GROUP BY report_date, SUM(stock_quantity * unit_cost)

### 4. `top 5 customers by total purchases`
**Type**: Customer entity, aggregation  
**Tests**: customer entity, limit handling, total_purchases metric  
**Expected**: *_customer_* views, ORDER BY total_purchases DESC, LIMIT 5

### 5. `sales revenue by category for Electronics in North region`
**Type**: Filters + dimension  
**Tests**: multiple filters, filter + dimension combination, category filter  
**Expected**: UNION ALL *_sales_* views, WHERE category='Electronics' AND region='North', GROUP BY category

### 6. `average order value by customer tier`
**Type**: Join query, calculated metric  
**Tests**: join handling (sales↔customer), calculated metric (avg_order_value), customer_tier dimension  
**Expected**: Join sales and customer views, GROUP BY customer_tier, calculate revenue/order_count

### 7. `products with low stock (below reorder level) in March`
**Type**: Filter on calculated/comparison  
**Tests**: comparison filters, inventory filters, single month selection  
**Expected**: mar012024_inventory_mar012024, WHERE stock_quantity < reorder_level

### 8. `revenue by region month over month`
**Type**: Multi-dimension trend (region + time)  
**Tests**: UNION ALL with multiple dimensions, time + region grouping, trend analysis  
**Expected**: UNION ALL *_sales_* views, GROUP BY report_date, region, SUM(quantity * price)

### 9. `sales revenue and order count by category`
**Type**: Multiple metrics, single dimension  
**Tests**: multiple metrics, metric aggregation (revenue + order_count), category dimension  
**Expected**: UNION ALL *_sales_* views, GROUP BY category, SUM(quantity * price) AS revenue, COUNT(DISTINCT order_id) AS order_count

### 10. `total stock value by supplier`
**Type**: Inventory aggregation, different dimension  
**Tests**: supplier dimension, stock_value metric, inventory-only query  
**Expected**: UNION ALL *_inventory_* views, GROUP BY supplier, SUM(stock_quantity * unit_cost)

---

## Test Coverage Matrix

| Query | Entity | Time | Dimensions | Metrics | Filters | Joins | UNION |
|-------|--------|------|------------|---------|---------|-------|-------|
| 1 | sales | multi | product | revenue | - | - | ✓ |
| 2 | sales | single | region, category | revenue | date | - | - |
| 3 | inventory | multi | report_date | stock_value | - | - | ✓ |
| 4 | customer | - | - | total_purchases | - | - | - |
| 5 | sales | - | category | revenue | category, region | - | ✓ |
| 6 | sales+customer | - | customer_tier | avg_order_value | - | ✓ | ✓ |
| 7 | inventory | single | - | - | comparison | - | - |
| 8 | sales | multi | region, report_date | revenue | - | - | ✓ |
| 9 | sales | - | category | revenue, order_count | - | - | ✓ |
| 10 | inventory | - | supplier | stock_value | - | - | ✓ |

---

## Expected Patterns to Verify

1. **aggregation_plan preservation**: Queries 1, 3, 5, 8, 9, 10 should have aggregation_plan set
2. **metric.definition usage**: All revenue queries should use `SUM(quantity * price)`, not `SUM(revenue)`
3. **UNION ALL for multi-month**: Queries 1, 3, 5, 8, 9, 10 should UNION views
4. **Single month optimization**: Query 2 should use specific view (jan012024_sales_jan012024)
5. **Join handling**: Query 6 should join sales and customer tables
6. **Multiple metrics**: Query 9 should generate both revenue and order_count

