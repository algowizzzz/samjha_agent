# Query Pattern Examples

## Example 1: Trend Analysis - Single Limit Over Time Period

**User Query:**
```
limit and exposure for limit id 300128 for past 7 days from 10 nov
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    exposure_amt,
    Original_limit,
    effective_limit,
    utilization
FROM limits_data
WHERE limit_id = 300128
  AND date >= DATE '2024-11-04'  -- 7 days back from Nov 10
  AND date <= DATE '2024-11-10'
ORDER BY date DESC
```

**Thinking to Build Query:**
- User wants trend analysis for a specific limit over a time period
- Filter by limit_id (300128) to get single limit
- Date range: past 7 days from Nov 10 = Nov 4-10
- Select exposure (exposure_amt) and limit columns (Original_limit, effective_limit)
- Include utilization to see capacity usage
- Order by date DESC to show most recent first

---

## Trend Analysis Examples

### Example 2: Trend Analysis - PV01 Limits Over Last 10 Days

**User Query:**
```
show me exposure and limit trends for all PV01 limits over the last 10 days
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE limit_type LIKE '%PV01%'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '9 days'
  AND date <= (SELECT MAX(date) FROM limits_data)
ORDER BY date DESC, utilization DESC
```

**Thinking to Build Query:**
- Filter by limit_type containing 'PV01' to get all PV01-related limits
- Use relative date calculation: MAX(date) - 9 days gives last 10 days
- Select exposure, limit, and utilization for trend analysis
- Order by date DESC (most recent first) and utilization DESC (highest usage first)

---

### Example 3: Trend Analysis - Specific Desk Utilization Trend

**User Query:**
```
what is the utilization trend for Canadian Options desk over the past week
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    limit_type,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE letter_nm = 'Canadian Options'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '6 days'
  AND date <= (SELECT MAX(date) FROM limits_data)
ORDER BY date DESC, limit_id
```

**Thinking to Build Query:**
- Filter by letter_nm = 'Canadian Options' to get specific trading desk
- Past week = last 7 days (MAX(date) - 6 days)
- Focus on utilization trend, so include exposure_amt, effective_limit, and utilization
- Order by date DESC to see progression, then by limit_id for consistency

---

### Example 4: Trend Analysis - High Utilization Limits Over Time

**User Query:**
```
show me limits that have been above 90% utilization for the past 5 days
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE utilization > 0.9
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '4 days'
  AND date <= (SELECT MAX(date) FROM limits_data)
ORDER BY date DESC, utilization DESC
```

**Thinking to Build Query:**
- Filter by utilization > 0.9 (90%) to find high-usage limits
- Past 5 days = MAX(date) - 4 days
- Include limit details (letter_nm, limit_type) to identify which limits
- Show exposure and effective_limit to see actual values
- Order by date DESC and utilization DESC to highlight most critical

---

### Example 5: Trend Analysis - Exposure Growth for Stress Limits

**User Query:**
```
how has exposure changed for stress limits over the last 2 weeks
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    exposure_amt,
    effective_limit,
    utilization,
    (exposure_amt - LAG(exposure_amt) OVER (PARTITION BY limit_id ORDER BY date)) AS exposure_change
FROM limits_data
WHERE limit_type LIKE '%Stress%'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '13 days'
  AND date <= (SELECT MAX(date) FROM limits_data)
ORDER BY date DESC, limit_id
```

**Thinking to Build Query:**
- Filter by limit_type containing 'Stress' to get stress limits
- Last 2 weeks = MAX(date) - 13 days (14 days total)
- Use LAG window function to calculate exposure change day-over-day
- Include exposure_amt, effective_limit, utilization for full picture
- Order by date DESC to see recent trends first

---

### Example 6: Trend Analysis - Capacity Usage by Limit Group

**User Query:**
```
show me the utilization trend for PV01 limit group over the past week, sorted by highest utilization
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE limit_group = 'PV01'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '6 days'
  AND date <= (SELECT MAX(date) FROM limits_data)
ORDER BY date DESC, utilization DESC
```

**Thinking to Build Query:**
- Filter by limit_group = 'PV01' to get all PV01 category limits
- Past week = MAX(date) - 6 days
- Select exposure, limit, and utilization for capacity analysis
- Order by date DESC (most recent first) and utilization DESC (highest first)

---

## Filter Examples

### Example 7: Filter by Trading Desk - Exposure View

**User Query:**
```
show me all exposure amounts and limits for Canadian Prime Brokerage and Margin Lending desk
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    limit_type,
    limit_group,
    exposure_amt,
    Original_limit,
    effective_limit,
    utilization
FROM limits_data
WHERE letter_nm = 'Canadian Prime Brokerage and Margin Lending'
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY utilization DESC
```

**Thinking to Build Query:**
- Filter by letter_nm (trading desk) for specific business unit
- Use latest date (MAX(date)) as default when not specified
- Focus on exposure (exposure_amt) and capacity (Original_limit, effective_limit)
- Include utilization to see which limits are most utilized
- Order by utilization DESC to highlight capacity concerns

---

### Example 8: Filter by Limit Type - Capacity View

**User Query:**
```
what are the limits and current exposure for all CVaR limits
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE limit_type LIKE '%CVaR%'
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY exposure_amt DESC
```

**Thinking to Build Query:**
- Filter by limit_type containing 'CVaR' to get all CVaR limits
- Default to latest date (MAX(date)) when not specified
- Show exposure_amt and effective_limit to see capacity vs usage
- Include utilization to understand capacity pressure
- Order by exposure_amt DESC to see largest exposures first

---

### Example 9: Filter by Measurement Unit - Multi-Currency Exposure

**User Query:**
```
show me all limits and exposures in JPY currency
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    meas_unit,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE meas_unit = 'JPY'
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY exposure_amt DESC
```

**Thinking to Build Query:**
- Filter by meas_unit = 'JPY' to get all JPY-denominated limits
- Include meas_unit in SELECT to confirm currency
- Show exposure_amt and effective_limit for capacity analysis
- Default to latest date when not specified
- Order by exposure_amt DESC to see largest JPY exposures

---

### Example 10: Filter by Limit Group - Risk Category View

**User Query:**
```
what is the exposure and capacity for all liquidity limits
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    limit_group,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE limit_group = 'Liquidity'
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY utilization DESC
```

**Thinking to Build Query:**
- Filter by limit_group = 'Liquidity' to get all liquidity risk limits
- Include limit_group in SELECT to show category
- Show exposure_amt and effective_limit for capacity vs usage
- Default to latest date when not specified
- Order by utilization DESC to identify limits approaching capacity

---

### Example 11: Filter by Aggregation Function - NET vs GROSS

**User Query:**
```
show me all gross limits and their current exposure
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    aggr_func_cd,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE aggr_func_cd LIKE '%GROSS%'
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY exposure_amt DESC
```

**Thinking to Build Query:**
- Filter by aggr_func_cd containing 'GROSS' to get gross aggregation limits
- Include aggr_func_cd in SELECT to show aggregation type
- Show exposure_amt and effective_limit for capacity analysis
- Default to latest date when not specified
- Order by exposure_amt DESC to see largest gross exposures

---

### Example 12: Filter by Date - Historical Snapshot

**User Query:**
```
show me all limits and exposures as of November 8th
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    letter_nm,
    limit_type,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE date = DATE '2024-11-08'
ORDER BY utilization DESC
```

**Thinking to Build Query:**
- Filter by specific date (DATE '2024-11-08') for historical snapshot
- Show exposure_amt and effective_limit for capacity analysis
- Include utilization to see capacity usage at that point in time
- Order by utilization DESC to identify limits with highest usage on that date

---

### Example 13: Multiple Filters - Combined Criteria

**User Query:**
```
show me NET PV01 limits for Canadian Options desk with utilization above 80%
```

**SQL Query:**
```sql
SELECT 
    date,
    limit_id,
    limit_type,
    limit_group,
    aggr_func_cd,
    exposure_amt,
    effective_limit,
    utilization
FROM limits_data
WHERE letter_nm = 'Canadian Options'
  AND limit_type LIKE '%PV01%'
  AND aggr_func_cd LIKE '%NET%'
  AND utilization > 0.8
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY utilization DESC
```

**Thinking to Build Query:**
- Multiple filters: letter_nm (desk), limit_type (PV01), aggr_func_cd (NET), utilization (>80%)
- Combine filters with AND to get precise subset
- Show exposure_amt and effective_limit for capacity analysis
- Default to latest date when not specified
- Order by utilization DESC to see most critical limits first

---

## Summary of Filter Patterns

### Filter Fields:
1. **letter_nm**: Trading desk or business unit (e.g., 'Canadian Options', 'US Delta One')
2. **limit_type**: Type of limit (e.g., 'PV01 Delta', 'CVaR / Limits', 'Gamma Vega')
3. **meas_unit**: Currency/unit of measurement (e.g., 'USD', 'JPY', 'CAD', 'EUR', 'MT', 'MWH')
4. **limit_group**: Risk category (e.g., 'PV01', 'Liquidity', 'Stress Limits', 'Issuer')
5. **date**: Specific time snapshot (e.g., DATE '2024-11-08' or MAX(date) for latest)
6. **aggr_func_cd**: Aggregation type (e.g., 'NET', 'GROSS', 'BY TRADE', 'GRID')

### Common Patterns:
- **Exposure View**: Focus on `exposure_amt` and `utilization`
- **Capacity View**: Focus on `effective_limit` and `utilization`
- **Trend Analysis**: Include date range and order by date
- **Default Date**: Use `(SELECT MAX(date) FROM limits_data)` when date not specified
- **High Utilization**: Filter by `utilization > 0.8` or `utilization > 0.9`
- **Multiple Filters**: Combine with AND for precise queries

