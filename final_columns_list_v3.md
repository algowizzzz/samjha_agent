# Final Column List for data_dictionary_risk_v3.json

## Complete Column List (41 columns) - FINAL VERSION

Based on all resolved contradictions and final business review:

### Core Data Fields (14 columns)
1. `date` - Date field (YYYY-MM-DD format)
2. `letter_nm` - Trading desk name
3. `limit_id` - Limit identifier
4. `id1` - Additional identifier field
5. `id2` - Limit identifier with class suffix
6. `limit_class` - Limit class (8 values: ex-ZNDRY, ex-PRIM, Inner RAL, Notice, Outer RAL, Primary, Reporting, Secondary)
7. `limit_group` - Risk category (78 values, letter-prefixed format, e.g., "A. VaR & Stress Limits")
8. `limit_type` - Specific limit type (122 values)
9. `limit_desc` - Limit description
10. `meas_unit` - Measurement unit (27 units: USD, CAD, EUR, etc.)
11. `aggr_func_cd` - Aggregation function code (32 valid values, "Pending Upload" is data quality issue)
12. `rating_floor` - Rating floor
13. `rating_ceiling` - Rating ceiling ✅ (included)
14. `unofficial_flag` - Unofficial flag indicator

### Limit & Exposure Fields (4 core columns)
15. `exposure_amt` - Current exposure amount
16. `Original_limit` - Original limit value
17. `effective_limit` - Effective limit value
18. `utilization` - Utilization percentage (exposure_amt / effective_limit)

### Grid & Reference Fields (4 columns)
19. `grid_cc_id` - Grid currency code identifier ✅ (included)
20. `grid_cell_id` - Grid cell identifier
21. `cur_id` - Currency identifier code ⚠️ (NOTE: use `cur_id` not `curr_id`)
22. `curr` - Currency code ✅ (included)

### Security & Issuer Fields (4 columns)
23. `sec_id` - Security identifier
24. `sec_desc` - Security description
25. `issuer_id` - Issuer/counterparty identifier
26. `issuer_nm` - Issuer name ⚠️ (NOTE: use `issuer_nm` not `issuer_mm`)

### Industry & Region Fields (4 columns)
27. `ind_class_id` - Industry classification identifier
28. `industry` - Industry name ✅ (NOTE: use `industry` not `ind_desc`)
29. `region_id` - Region identifier code
30. `region_cd` - Region code

### Workflow & Status Fields (3 columns)
31. `wf_instance_id` - Workflow/approval instance identifier ⚠️ (NOTE: use `wf_instance_id` not `wl_instance_id`)
32. `state` - State values: ["Approved", "Pending", "Rejected"]
33. `extension` - Extended total limit value (when extension is active)

### Date Fields (2 columns)
34. `st_dt` - Start date ⚠️ (NOTE: use `st_dt` not `start_date`)
35. `end_dt` - End date

### Display & Metadata Fields (5 columns)
36. `limit_concatenate` - Concatenated limit identifier for display ⚠️ (NOTE: use `limit_concatenate` not `mtfi_concatenate`)
37. `pref_name` - Preferred/display name
38. `LM_Column1` - Additional metadata column 1 ⚠️ (NOTE: prefix is `LM_` from CSV)
39. `LM_Column2` - Additional metadata column 2
40. `LM_Column3` - Additional metadata column 3
41. `LM_Column4` - Additional metadata column 4

**Total: 41 columns**

---

## Fields EXCLUDED from v3

Based on business review, these fields should NOT be included:

1. ❌ `ind_desc` - Use `industry` instead (they are the same)

---

## Fields from Business Samples that are IGNORED

These appeared in business samples but are NOT real fields:

1. ❌ `date_mm` - Not a real field (only `date` exists)
2. ❌ `inst_id` - Not a real field (use `wf_instance_id` instead)
3. ❌ `inst_instance_id` - Not a real field (use `wf_instance_id` instead)
4. ❌ `limit_usd` - Ignore - do not add
5. ❌ `mtfi_concatenate` - Not a real field (use `limit_concatenate` instead)
6. ❌ `issuer_mm` - Use `issuer_nm` instead

---

## Field Name Corrections Applied

| CSV Column Name | Correct Name for v3 | Status |
|-----------------|---------------------|--------|
| `wl_instance_id` | `wf_instance_id` | ✅ Changed |
| `curr_id` | `cur_id` | ✅ Changed |
| `issuer_mm` (from samples) | `issuer_nm` | ✅ Use CSV name |
| `start_date` (from samples) | `st_dt` | ✅ Use CSV name |
| `mtfi_concatenate` (from samples) | `limit_concatenate` | ✅ Use CSV name |
| `means_unit` (typo in reference) | `meas_unit` | ✅ Use CSV name |
| `IM_Column1-4` (from samples) | `LM_Column1-4` | ✅ Use CSV prefix |
| `ind_desc` (from samples) | `industry` | ✅ Use CSV name |

---

## Summary by Category

| Category | Count | Columns |
|----------|-------|---------|
| Core Data | 14 | date, letter_nm, limit_id, id1, id2, limit_class, limit_group, limit_type, limit_desc, meas_unit, aggr_func_cd, rating_floor, rating_ceiling, unofficial_flag |
| Limit & Exposure | 4 | exposure_amt, Original_limit, effective_limit, utilization |
| Grid & Reference | 4 | grid_cc_id, grid_cell_id, cur_id, curr |
| Security & Issuer | 4 | sec_id, sec_desc, issuer_id, issuer_nm |
| Industry & Region | 4 | ind_class_id, industry, region_id, region_cd |
| Workflow & Status | 3 | wf_instance_id, state, extension |
| Dates | 2 | st_dt, end_dt |
| Display & Metadata | 5 | limit_concatenate, pref_name, LM_Column1-4 |
| **TOTAL** | **41** | |

---

## Core Fields Confirmed ✅

These 4 fields are core and must be properly defined in v3:

1. ✅ `exposure_amt` - Current exposure amount
2. ✅ `Original_limit` - Original limit value  
3. ✅ `effective_limit` - Effective limit value
4. ✅ `utilization` - Utilization percentage

---

## Final Changes from Previous Version

1. ✅ **Added:** `rating_ceiling` - Rating ceiling (now included)
2. ✅ **Added:** `grid_cc_id` - Grid currency code identifier (now included)
3. ✅ **Added:** `curr` - Currency code (now included)
4. ✅ **Changed:** `ind_desc` → `industry` (use industry field name)
5. ✅ **Confirmed:** `wf_instance_id` - Correct field name
6. ✅ **Confirmed:** `limit_concatenate` - Correct field name

---

*Last Updated: Final business review completed - Ready for v3 JSON creation*
