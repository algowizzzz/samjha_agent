# Common user questions (with likely follow-ups)

1. Show me the top limits.
   — Follow-ups: “Top by what metric?” / “Top N (5/10/20)?” / “Only Primary?” / “For latest date or range?”

2. Show me top PV01 limits.
   — Follow-ups: “Top 10?” / “By desk or overall?” / “Include Secondary?” / “Show exposure and utilization?”

3. How many of these are for Energy?
   — Follow-ups: “Do you mean the list I just showed?” / “Count by currency?” / “List the IDs or desks?”

4. What is the sum of exposure across these limits?
   — Follow-ups: “Sum by currency or converted?” / “Include extensions?” / “Show breakdown by desk?”

5. Show me limits with utilization > 0.95.
   — Follow-ups: “Do you mean >= 0.95 or > 0.95?” / “Limit to Primary only?” / “Sort by exposure or utilization?”

6. Which limits are breached?
   — Follow-ups: “Show breached with or without extension?” / “Include end_dt for extensions?” / “Notify historical breach frequency?”

7. Show trend for this limit over the last 7 days.
   — Follow-ups: “Use limit_id or letter_nm + type?” / “Include exposure change vs prior day?” / “Return chart or table?”

8. Show me PV01 limits for Canadian desks.
   — Follow-ups: “Exact desk names or pattern match?” / “Latest snapshot or past week?” / “Top N by utilization?”

9. How many primary limits are in breach by industry?
   — Follow-ups: “Group by industry only or also by region?” / “Show counts and total exposure?” / “Include near-breach (>=0.9)?”

10. Show me extensions expiring within 7 days.
    — Follow-ups: “Only Active state?” / “Provide owner/contact?” / “Include utilization and exposure?”

11. Give me average utilization by limit_group for the past week.
    — Follow-ups: “Include count of limits per group?” / “Exclude Decommissioned?” / “Return CSV?”

12. Show me limits that were above 90% for the last 5 days.
    — Follow-ups: “Require consecutive days or any occurrence?” / “Return list of limit_ids?” / “Include trend sparkline?”

13. Show me limits for Canadian Options desk.
    — Follow-ups: “All limit_groups or a specific one?” / “Latest only or historical?” / “Include limit_type detail?”

14. What changed since yesterday for this limit?
    — Follow-ups: “Which fields to compare (exposure, effective_limit, utilization)?” / “Show percentage change?” / “Show prior-day SQL used?”

15. Show me the top breached limits without extensions.
    — Follow-ups: “Sort by utilization or exposure?” / “Limit to Primary?” / “Include issuer info?”

16. Convert totals to CAD.
    — Follow-ups: “Which FX rate (spot, mid, latest)?” / “Convert all currencies or only USD/EUR?” / “Show conversion assumptions?”

17. Show me NET PV01 limits only.
    — Follow-ups: “Match aggr_func_cd LIKE '%NET%'?” / “Also include NET SHORT BY SEC?” / “Sort by utilization?”

18. How many limits did not change in the last 7 days?
    — Follow-ups: “Define ‘did not change’ as same effective_limit or same utilization?” / “Return list or count only?”

19. Show me limits with missing/invalid data.
    — Follow-ups: “Which columns to validate?” / “Return sample rows and error type?” / “Prioritise high-utilization issues?”

20. Show me average exposure by region for Stress Limits.
    — Follow-ups: “Group by date or aggregate over period?” / “Include count of limits per region?” / “Limit to Primary?”

21. Which desks have the most limits near breach?
    — Follow-ups: “Define near breach (>=0.9 or >=0.95)?” / “Return top N desks?” / “Include total exposure per desk?”

22. Show me limits where meas_unit is physical (BBL, MT).
    — Follow-ups: “Filter by specific unit?” / “Aggregate only within same unit?” / “Convert to common unit?”

23. Show me the SQL you used.
    — Follow-ups: “Include applied defaults and inherited context?” / “Show alternative SQL if no rows returned?”

24. Why did I get zero rows for my query?
    — Follow-ups: “Show relaxed query suggestions?” / “Suggest closest matches (limit_type vs limit_group)?” / “Run fuzzy match?”

25. Alert me when any limit breaches.
    — Follow-ups: “Frequency of checks (real-time/daily)?” / “Channels (email/Slack)?” / “Include top offending limits only?”

26. Show me exposure change for PV01 limits vs prior day.
    — Follow-ups: “Absolute or percent change?” / “Include only Primary?” / “Show top increases/decreases?”

27. Show me limits by issuer for CVaR limits.
    — Follow-ups: “Group by issuer_nm or issuer_id?” / “Filter by region or currency?” / “Sort by total exposure?”

28. Which limits were decommissioned recently?
    — Follow-ups: “Define ‘recently’ (N days)?” / “Show reason or state transitions?” / “Include prior exposure and date of decommission?”

29. Show me all limits that include 'stress' in description.
    — Follow-ups: “Search limit_type vs limit_desc?” / “Return exact matches or ILIKE %stress%?” / “Limit by date?”

30. I want a short summary of risk hotspots.
    — Follow-ups: “Define timeframe and thresholds?” / “Deliverable format (one-pager, table, dashboard widgets)?” / “Include suggested actions or owners?”

---
Below is a structured list of **user question patterns** specifically focused on *arithmetic / quantitative follow-ups*.
These represent how real users think, talk, and request calculations when they see numbers in the table.

---

## ✅ Arithmetic / Aggregation Requests (direct on result set)

| Category       | Example user prompt (follow-up)                       | Computation needed                  |
| -------------- | ----------------------------------------------------- | ----------------------------------- |
| Sum / Total    | “What is the **total exposure** across these limits?” | `SUM(exposure_amt)`                 |
| Average        | “What is the **average utilization** of these?”       | `AVG(utilization)`                  |
| Max / Highest  | “Which one has the **highest exposure**?”             | `MAX(exposure_amt)` + TOP 1         |
| Min / Lowest   | “Show the **lowest utilization** among these.”        | `MIN(utilization)`                  |
| Median         | “What’s the **median exposure** here?”                | `PERCENTILE_CONT(0.5)`              |
| Count          | “How many **limits are breached**?”                   | `COUNT(*) WHERE utilization >= 1.0` |
| Count distinct | “How many **desks** are represented in these limits?” | `COUNT(DISTINCT letter_nm)`         |
| Ratio          | “What percent of these are breached?”                 | `(COUNT(breached)/COUNT(all))*100`  |

---

## ✅ Trend / Time Arithmetic

| Category                  | Example user prompt                                            | Computation / SQL concept                                      |
| ------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| Change vs yesterday       | “How much has exposure **changed since yesterday**?”           | `exposure_amt - LAG(exposure_amt)`                             |
| Percentage change         | “% change day over day?”                                       | `(current - prior) / prior`                                    |
| Days in breach            | “How many **days was this limit in breach** in the last week?” | `COUNT(*) WHERE utilization >= 1.0`                            |
| Consecutive breach streak | “Has this limit been in breach for **3 days in a row**?”       | window + sequence check                                        |
| Average over time         | “What’s the **7-day average exposure**?”                       | windowed `AVG(exposure_amt)`                                   |
| Moving average            | “Show the **rolling moving average** for utilization.”         | window function: `AVG() OVER (ORDER BY date ROWS 3 PRECEDING)` |

---

## ✅ Arithmetic by Group / Category

| Category           | Example user prompt                        | Requires…                             |
| ------------------ | ------------------------------------------ | ------------------------------------- |
| Group sum          | “Sum the exposure **by desk**.”            | `GROUP BY letter_nm`                  |
| Group avg          | “Average utilization **per industry**.”    | `GROUP BY industry`                   |
| Group distribution | “Show the **exposure split** by currency.” | `GROUP BY meas_unit`                  |
| Ranking            | “Top 3 industries by total exposure.”      | `SUM(exposure_amt)` + `ORDER BY DESC` |
| Compare two groups | “Compare PV01 vs Stress limits.”           | side-by-side aggregation              |

---

## ✅ Capacity / Remaining / Derived Metrics

| Category                     | Example user prompt                                     | Computation                      |
| ---------------------------- | ------------------------------------------------------- | -------------------------------- |
| Remaining capacity           | “How much **capacity is left**?”                        | `effective_limit - exposure_amt` |
| Remaining %                  | “What percent **capacity remains**?”                    | `1 - utilization`                |
| Exposure to limit ratio      | “What’s the **exposure-to-limit ratio**?”               | `exposure_amt / effective_limit` |
| Risk sensitivity adjustments | “If exposure increases by 10%, what’s new utilization?” | `utilization * 1.10`             |

---

## ✅ What-If arithmetic (simulation)

| Category            | Example user prompt                            | Computation model                        |
| ------------------- | ---------------------------------------------- | ---------------------------------------- |
| Scenario multiplier | “If exposure increases 20%, will we breach?”   | `(exposure_amt * 1.2) / effective_limit` |
| Limit change        | “If we increase limit by 5M, new utilization?” | `exposure_amt / (effective_limit + 5M)`  |
| FX conversion       | “Convert totals to CAD.”                       | apply FX mapping                         |

---

## ✅ Ranking & Sorting Arithmetic (comparative)

| Category                   | Example follow-up                  | Computation                         |
| -------------------------- | ---------------------------------- | ----------------------------------- |
| Top N based on utilization | “Top 3 by utilization.”            | `ORDER BY utilization DESC LIMIT 3` |
| Bottom N                   | “Which limits are least utilized?” | `ORDER BY utilization ASC`          |
| Sort by exposure growth    | “Sort by increasing exposure.”     | `ORDER BY exposure_amt - LAG()`     |

---

## ✅ Statistical / KPI driven

| Category              | User prompt                                                        | Computation             |
| --------------------- | ------------------------------------------------------------------ | ----------------------- |
| Distribution shapes   | “Show histogram buckets of utilization 0–50, 50–80, 80–100, >100.” | bucket function         |
| Percentiles           | “95th percentile exposure across these.”                           | `PERCENTILE_CONT(0.95)` |
| Variance / volatility | “Which limit is most volatile over last 5 days?”                   | `STDDEV(exposure_amt)`  |

---

### What this means for your agent design

Users don’t think in SQL;
**they think in business arithmetic over already seen numbers.**

Your agent must:

1. Detect aggregation intent (sum/avg/max/min/count).
2. Apply it to the **previous result set, not the full table**.
3. Resolve pronouns (“these”, “those”, “the above”).

---

# Examples — minimal-filter journeys to identify a unique limit

Below are **6 production-ready examples** (user utterance → node flow → minimal filters used → SQL template → clarifying follow-ups and fallback logic). Use these as explicit cases your agent can match to reduce ambiguity and speed to answer.

---

## Example 1 — User supplies the unique ID (best case)

**User:**
“Show me limit_id 300025.”

**Node flow:**
`invoke → check_structure → check_ambiguity (none) → generate_sql → execute → synthesize → end`

**Minimal filters used:**
`limit_id`

**SQL template:**

```sql
SELECT *
FROM limits_data
WHERE limit_id = 300025
  AND date = (SELECT MAX(date) FROM limits_data); -- default date
```

**Agent behavior / follow-ups:**
No clarifier required. Show full row + provenance. Offer related actions: “Show 7-day trend?” or “Convert exposure to CAD?”

**Fallback:**
If 0 rows, reply: “I couldn't find limit_id 300025 in the latest snapshot — do you want me to search historical dates or fuzzy-match similar IDs?”

---

## Example 2 — Desk + metric + type + unit (typical unique)

**User:**
“Show me the Canadian Options PV01 limit in CAD.”

**Node flow:**
`invoke → check_structure (extracts letter_nm, limit_group, meas_unit) → check_ambiguity (likely unique) → generate_sql → execute → synthesize → end`

**Minimal filters used:**
`letter_nm`, `limit_group`, `limit_type` *if provided*, `meas_unit`, `date (default)`

**SQL template:**

```sql
SELECT *
FROM limits_data
WHERE letter_nm = 'Canadian Options'
  AND limit_group = 'PV01'
  AND meas_unit = 'CAD'
  AND date = (SELECT MAX(date) FROM limits_data);
```

*(If `limit_type` provided, add `AND limit_type = 'PV01 Delta'`.)*

**Agent behavior / follow-ups:**
If single row — present row and ask if user wants trend/owners. If multiple rows — ask: “I found N matches. Do you mean `PV01 Delta` vs `PV01 Stress` or should I include other currencies?”

**Fallback:**
If multiple rows, propose the top 3 `limit_id`s for confirmation.

---

## Example 3 — Desk + limit_group only (ambiguous; needs one clarifier)

**User:**
“Show me the PV01 limit for SET Management.”

**Node flow:**
`invoke → check_structure → check_ambiguity (multiple matches possible) → process_clarification → generate_sql → execute → synthesize → end`

**Minimal filters detected:**
`letter_nm`, `limit_group` — insufficient to guarantee uniqueness

**Clarifying question (agent asks):**
“Do you mean `limit_type = 'Gamma Vega'` or `limit_type = 'PV01 Delta'`, or should I show all PV01-related rows for SET Management?”

**On user reply (example: user says “PV01 Delta”): SQL:**

```sql
SELECT *
FROM limits_data
WHERE letter_nm = 'SET Management'
  AND limit_group = 'PV01'
  AND limit_type = 'PV01 Delta'
  AND date = (SELECT MAX(date) FROM limits_data);
```

**Fallback if user says “show all”:**
Return all matches, sorted `ORDER BY utilization DESC`, and show `limit_id` for selection.

---

## Example 4 — Metric + partial desk name (fuzzy match)

**User:**
“Top PV01 limits for Canadian desks”
(likely multiple desks matching ‘Canadian’)

**Node flow:**
`invoke → check_structure → check_ambiguity (desk pattern) → process_clarification (optional) → generate_sql → execute → synthesize → end`

**Minimal filters used (inferred):**
`limit_group = 'PV01'`, `letter_nm ILIKE '%Canadian%'`, `limit_class = 'Primary'`, `date = MAX(date)`

**SQL template:**

```sql
SELECT date, limit_id, letter_nm, limit_type, exposure_amt, effective_limit, utilization
FROM limits_data
WHERE limit_group = 'PV01'
  AND letter_nm ILIKE '%Canadian%'
  AND limit_class = 'Primary'
  AND date = (SELECT MAX(date) FROM limits_data)
ORDER BY utilization DESC
LIMIT 10;
```

**Clarifying question (optional):**
“Do you want exact matches only (e.g., 'Canadian Options') or all desks with 'Canadian' in the name?”

**Fallback / Explanation:**
If user prefers exact match, switch to `letter_nm = 'Canadian Options'` and re-run.

---

## Example 5 — Time-specific request (user gives date)

**User:**
“Show me the PV01 limit for GM Opportunity Trading on 2024-10-11.”

**Node flow:**
`invoke → check_structure (extract date) → check_ambiguity → generate_sql → execute → synthesize → end`

**Minimal filters used:**
`letter_nm`, `limit_group`, `date` (explicit supplied), optionally `limit_type`/`meas_unit`

**SQL template:**

```sql
SELECT *
FROM limits_data
WHERE letter_nm = 'GM Opportunity Trading'
  AND limit_group = 'PV01'
  AND date = DATE '2024-10-11';
```

**Agent behavior / follow-ups:**
Return the historical snapshot; offer trend or “latest snapshot” comparison. If multiple rows for that date, ask for `meas_unit` or `limit_type`.

---

## Example 6 — User gives derived description (text search)

**User:**
“Find the limit described ‘CRUDE_OIL_PV01_EU’.”

**Node flow:**
`invoke → check_structure (detects text in limit_desc/pref_name) → check_ambiguity → generate_sql → execute → synthesize → end`

**Minimal filters used:**
`limit_desc ILIKE '%CRUDE_OIL_PV01_EU%'` (or `pref_name`), `date = MAX(date)`

**SQL template:**

```sql
SELECT *
FROM limits_data
WHERE limit_desc ILIKE '%CRUDE_OIL_PV01_EU%'
  AND date = (SELECT MAX(date) FROM limits_data);
```

**Clarifying question (if multiple matches):**
“I found N descriptions with that phrase — do you mean the one for 'Oil Products NGL Trading' (limit_id 300001) or another?”

---

# Compact decision rules (to encode into `clarify_node`)

1. **If `limit_id` present → use it (no clarifier).**
2. **Else require `letter_nm + limit_group`.**

   * If that returns >1 row → ask for `limit_type` **or** `meas_unit`.
   * If user prefers fuzzy desk names, allow `ILIKE` and confirm.
3. **If user supplies explicit `date` → use it; otherwise default to `date = MAX(date)`.**
4. **Apply defaults automatically:** `limit_class = 'Primary'` and `state != 'Decommissioned'` unless user states otherwise.
5. **When aggregations or conversions are requested, surface `meas_unit` differences** before summing.

# Suggested short clarifying questions (use these verbatim)

* “Do you have the `limit_id`, or should I find by desk and metric?”
* “I found multiple matches — do you mean `limit_type = 'PV01 Delta'` or `PV01 Stress`?”
* “Should I use exact desk name `Canadian Options` or include partial matches (`%Canadian%`)?”
* “These rows include multiple currencies — sum by currency or convert to CAD?”

---

If you want, I can now:

* produce a CSV/JSON mapping of these examples for direct ingestion by your planner, **or**
* turn the decision rules into pseudocode for your `clarify_node`.

Which would you like next?
