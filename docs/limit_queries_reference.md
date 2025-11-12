# Limit Queries SQL Reference Guide

This document contains SQL queries for common limit analysis tasks based on the `limits_data.csv` file. Each query includes a brief explanation for senior data analysts.

**Note:** When using DuckDB, the table name will be `limits_data` (derived from the CSV filename). The data spans from 2024-10-09 to 2024-11-08.

---

## Common Queries

### Q1: Show me latest data

**Query:**
```sql
SELECT *
FROM limits_data
WHERE date = (SELECT MAX(date) FROM limits_data)
ORDER BY limit_id, date DESC;
```

**Explanation:** Retrieves all records for the most recent date in the dataset by using a subquery to find the maximum date, then filtering and ordering by limit_id and date.

---

### Q2: Show me breached limits (utilization >= 1.0)

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr,
    state
FROM limits_data
WHERE utilization >= 1.0
ORDER BY utilization DESC, date DESC;
```

**Explanation:** Filters records where utilization is greater than or equal to 1.0 (100%), indicating the limit has been breached, and orders by utilization severity and date.

---

### Q3: Show me limits near breach (utilization >= 0.9)

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr,
    state
FROM limits_data
WHERE utilization >= 0.9 AND utilization < 1.0
ORDER BY utilization DESC, date DESC;
```

**Explanation:** Identifies limits at risk of breach by filtering utilization between 0.9 (90%) and 1.0, excluding already breached limits, sorted by proximity to breach threshold.

---

### Q4: Show me PV01 limits for Canadian Options

**Query:**
```sql
SELECT 
    date,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr,
    state
FROM limits_data
WHERE letter_nm = 'Canadian Options'
  AND limit_type = 'PV01 Delta'
ORDER BY date DESC, limit_id;
```

**Explanation:** Filters by desk name 'Canadian Options' and limit type 'PV01 Delta' to retrieve all PV01-related limits for this specific desk, ordered chronologically.

---

### Q5: Show me limits with active extensions

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    extension,
    st_dt,
    end_dt,
    state
FROM limits_data
WHERE extension = 1
ORDER BY date DESC, limit_id;
```

**Explanation:** Returns all limits where the extension flag equals 1, indicating an active extension is in place, including extension start and end dates for reference.

---

### Q6: Show me Primary limits only

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    id2,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    state
FROM limits_data
WHERE id2 LIKE '%-PRIMARY'
ORDER BY date DESC, limit_id;
```

**Explanation:** Filters records where the id2 column contains '-PRIMARY' suffix, identifying primary limits (as opposed to secondary limits), using LIKE pattern matching for flexibility.

---

## Trend Analysis Queries

### Q7: Show me Canadian Options PV01 limit for the past 7 days

**Query:**
```sql
SELECT 
    date,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr
FROM limits_data
WHERE letter_nm = 'Canadian Options'
  AND limit_type = 'PV01 Delta'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '7 days'
ORDER BY date DESC, limit_id;
```

**Explanation:** Retrieves PV01 Delta limits for Canadian Options desk over the last 7 days using date arithmetic relative to the maximum date in the dataset, providing a week-over-week trend view of limit utilization.

---

### Q8: What is the SET Management Gamma limit for the past week

**Query:**
```sql
SELECT 
    date,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr
FROM limits_data
WHERE letter_nm = 'SET Management'
  AND limit_type = 'Gamma Vega'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '7 days'
ORDER BY date DESC;
```

**Explanation:** Filters for SET Management desk's Gamma Vega limits over the past 7 days relative to the latest date in the dataset, showing daily exposure and utilization trends for gamma risk limits.

---

### Q9: Show me limit and exposure for limit id 300128 for past 7 days

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    limit_desc,
    exposure_amt,
    Original_limit,
    effective_limit,
    utilization,
    curr,
    state
FROM limits_data
WHERE limit_id = 300128
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '7 days'
ORDER BY date DESC;
```

**Explanation:** Provides a historical view of a specific limit (300128) showing both original and effective limits alongside actual exposure over the past week relative to the latest date, for trend analysis.

---

### Q10: Show me Primary PV01 limits for Canadian Options desk for the latest 7 days

**Query:**
```sql
SELECT 
    date,
    limit_id,
    id2,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr
FROM limits_data
WHERE letter_nm = 'Canadian Options'
  AND limit_type = 'PV01 Delta'
  AND id2 LIKE '%-PRIMARY'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '7 days'
ORDER BY date DESC, limit_id;
```

**Explanation:** Combines multiple filters (desk, limit type, primary status, and date range) to show only primary PV01 limits for Canadian Options over the past week relative to the latest date, excluding secondary limits.

---

## Complex Queries

### Q11: Show me Canadian PV01 limits over the past 2 weeks

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr
FROM limits_data
WHERE letter_nm LIKE '%Canadian%'
  AND limit_type = 'PV01 Delta'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '14 days'
ORDER BY letter_nm, date DESC, limit_id;
```

**Explanation:** Uses LIKE pattern matching to capture all Canadian desks (e.g., Canadian Options, Canadian Prime Finance) and filters PV01 Delta limits over 14 days relative to the latest date, grouped by desk for comparison.

---

### Q12: Show me limits that breached in the last 10 days

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    curr,
    state
FROM limits_data
WHERE utilization >= 1.0
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '10 days'
ORDER BY date DESC, utilization DESC;
```

**Explanation:** Identifies all breached limits (utilization >= 1.0) within the last 10 days relative to the latest date, ordered by date and severity, to track recent limit violations for risk management review.

---

### Q13: How has exposure changed for stress limits in CAD currency over the past week

**Query:**
```sql
SELECT 
    date,
    letter_nm,
    limit_id,
    limit_desc,
    exposure_amt,
    effective_limit,
    utilization,
    LAG(exposure_amt) OVER (PARTITION BY limit_id ORDER BY date) AS previous_exposure,
    exposure_amt - LAG(exposure_amt) OVER (PARTITION BY limit_id ORDER BY date) AS exposure_change
FROM limits_data
WHERE limit_group = 'Stress Limits'
  AND curr = 'CAD'
  AND date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '7 days'
ORDER BY limit_id, date;
```

**Explanation:** Uses window functions (LAG) to calculate day-over-day exposure changes for stress limits in CAD, showing both current exposure and the delta from the previous day for trend analysis over the past week.

---

### Q14: Show me average utilization by limit group over the past week

**Query:**
```sql
SELECT 
    limit_group,
    COUNT(DISTINCT limit_id) AS unique_limits,
    AVG(utilization) AS avg_utilization,
    MAX(utilization) AS max_utilization,
    MIN(utilization) AS min_utilization
FROM limits_data
WHERE date >= (SELECT MAX(date) FROM limits_data) - INTERVAL '7 days'
GROUP BY limit_group
ORDER BY avg_utilization DESC;
```

**Explanation:** Aggregates utilization metrics by limit_group using GROUP BY, calculating average, max, and min utilization along with count of unique limits to identify which limit groups are most utilized over the past week.

---

### Q15: How many limits are breached?

**Query:**
```sql
SELECT 
    COUNT(DISTINCT limit_id) AS breached_limit_count,
    COUNT(*) AS total_breach_records
FROM limits_data
WHERE utilization >= 1.0
  AND date = (SELECT MAX(date) FROM limits_data);
```

**Explanation:** Counts distinct breached limits and total breach records for the latest date, providing both unique limit count and total breach instances (accounting for multiple dates per limit).

---

### Q16: Total exposure by desk

**Query:**
```sql
SELECT 
    letter_nm,
    COUNT(DISTINCT limit_id) AS limit_count,
    SUM(exposure_amt) AS total_exposure,
    AVG(utilization) AS avg_utilization,
    MAX(utilization) AS max_utilization
FROM limits_data
WHERE date = (SELECT MAX(date) FROM limits_data)
GROUP BY letter_nm
ORDER BY total_exposure DESC;
```

**Explanation:** Aggregates exposure metrics by desk for the latest date, showing total exposure, limit count, and utilization statistics to identify desks with highest exposure and risk concentration.

---

## Additional Utility Queries

### Get all unique desks

**Query:**
```sql
SELECT DISTINCT letter_nm
FROM limits_data
ORDER BY letter_nm;
```

**Explanation:** Returns a list of all unique desk names in the dataset for reference when building desk-specific queries.

---

### Get all limit types

**Query:**
```sql
SELECT DISTINCT limit_type
FROM limits_data
ORDER BY limit_type;
```

**Explanation:** Lists all distinct limit types available in the dataset (e.g., PV01 Delta, Gamma Vega, Stress Limits) for query construction reference.

---

### Date range in dataset

**Query:**
```sql
SELECT 
    MIN(date) AS earliest_date,
    MAX(date) AS latest_date,
    COUNT(DISTINCT date) AS total_days
FROM limits_data;
```

**Explanation:** Provides the date range and total number of distinct dates in the dataset, useful for validating date filters in trend queries.

---

## Notes for Data Analysts

1. **Date Handling:** The queries use `(SELECT MAX(date) FROM limits_data) - INTERVAL 'N days'` syntax which is DuckDB compatible and works relative to the actual data in the table, not the current system date. This ensures queries work correctly even with historical datasets.

2. **Table Name:** When loading the CSV into DuckDB, the table name will be `limits_data` (derived from filename). If using a different name, update all queries accordingly.

3. **Utilization Calculation:** Utilization is already calculated in the dataset as `exposure_amt / effective_limit`. The queries filter on this pre-calculated value.

4. **Primary vs Secondary:** The `id2` column contains values like '300001-PRIMARY' or '300002-SECONDARY'. Use pattern matching with LIKE to filter.

5. **Extension Flag:** The `extension` column is binary (0 or 1), where 1 indicates an active extension.

6. **Performance:** For large datasets, consider adding indexes on frequently filtered columns (date, limit_id, letter_nm, limit_type) if your database supports them.

7. **Column Names:** All column names match exactly as they appear in the CSV header. Note that `Original_limit` has a capital 'O' and `effective_limit` is lowercase.

8. **Date Range:** The current dataset spans from 2024-10-09 to 2024-11-08. Queries using relative date intervals (e.g., "past 7 days") will work correctly as long as there is sufficient historical data in the table.

---

*Last Updated: Based on limits_data.csv schema verification*

