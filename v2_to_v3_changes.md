# Data Dictionary Update: v2 → v3 Change Document

## Summary
This document details all changes needed to update `data_dictionary_risk_v2.json` to `data_dictionary_risk_v3.json` based on new business-provided data details from `new_limits_data_details.json`.

**Note:** Following business user guidance - using details from `new_limits_data_details.json` as the source of truth, not CSV headers.

## ✅ Key Resolutions (All Contradictions Resolved - FINAL)

### Final Field List: **41 columns total**

1. **Date format:** YYYY-MM-DD (e.g., "2024-10-09") - only `date` field exists
2. **Workflow ID:** Use `wf_instance_id` 
3. **Extension field:** Contains value for the extended total limit
4. **State values:** `["Approved", "Pending", "Rejected"]`
5. **Field names:** `meas_unit`, `limit_concatenate`, `st_dt`, `cur_id`, `issuer_nm`, `industry`
6. **Limit group format:** Letter-prefixed from business reference (e.g., "A. VaR & Stress Limits")
7. **Core fields to include:** `exposure_amt`, `Original_limit`, `effective_limit`, `utilization`
8. **Additional fields included:** `rating_ceiling`, `grid_cc_id`, `curr`, `industry`
9. **Data quality:** "Pending Upload" in aggr_func_cd is a data quality issue - not a valid value
10. **Fields to exclude:** `limit_usd`, `ind_desc` (use `industry` instead)

---

## Section 1: Update Reference Values (data_samples)

**Location:** `procedural_knowledge.data_samples`

### 1.1 Update `typical_letter_nm_values`
- **Current (v2):** 8 examples: `["Oil Products NGL Trading", "US Delta One", "Canadian Options", ...]`
- **Change:** Replace with full list from `new_limits_data_details.json.reference_values.letter_nm` (129 desk names)
- **Source:** Lines 3-129 in new file
- **Action:** Replace entire array with all 129 desk names

### 1.2 Update `typical_limit_groups`
- **Current (v2):** `["PV01", "RR & Gamma", "Asset", "Stress Limits", "Liquidity", "Notional", "Issuer"]`
- **Change:** Replace with full list from `reference_values.limit_group` (78 values)
- **Note:** Many start with letters (A., B., C., etc.) - e.g., "A. VaR & Stress Limits", "C. PV01 Curve Risk Limits"
- **Source:** Lines 140-218 in new file
- **Action:** Replace entire array with all 78 limit group values

### 1.3 Update `typical_limit_classes`
- **Current (v2):** `["Primary", "Secondary", "Notice", "Reporting", "ex-PRIM", "ex-ZNDRY"]`
- **Change:** Update to: `["ex-ZNDRY", "ex-PRIM", "Inner RAL", "Notice", "Outer RAL", "Primary", "Reporting", "Secondary"]`
- **Source:** Lines 130-139 in new file
- **Action:** Replace with exact list from business details

### 1.4 Add `typical_limit_types`
- **Current (v2):** Not present
- **Change:** Add new field with examples from `reference_values.limit_type` (122 values total)
- **Examples to include:** First 15-20 values like `["PV01", "EQ VEGA", "IR GAMMA", "Base PV01", "CR01", "Stress", "PV01 Delta", "Gamma Vega", ...]`
- **Source:** Lines 219-341 in new file
- **Action:** Add new field with representative examples (top 20)

### 1.5 Update `typical_aggr_func_cd`
- **Current (v2):** Generic examples in column definition only
- **Change:** Add comprehensive list to data_samples section (32 total values)
- **Examples:** `["GRID", "GROSS", "NET", "GROSS BY SECURITY", "SINGLE NAME", "SINGLE INDUSTRY", "NET LONG BY SEC", "BY TRADE", ...]`
- **Source:** Lines 342-374 in new file
- **Action:** Add new field with all 32 aggregation function codes

### 1.6 Update `meas_unit` examples
- **Current (v2):** `"Currency/unit (USD, CAD, EUR, JPY, BBL, OZ, MT, MWH)"`
- **Change:** Expand to include all units from `reference_values.means_unit` (27 values)
- **New units to add:** AUD, BPS, BU, CHF, CON, CREDIT, D, DAYS, Event, FUT/EQVN, GUAL, GBP, LIB, MXN, NZD, Percent_Rating_ST, UKN, Y,ZAR
- **Note:** Business details use field name `means_unit` in reference, but sample records use `meas_unit` - use `meas_unit`
- **Source:** Lines 375-403 in new file
- **Action:** Update column definition and add to data_samples

---

## Section 2: Add Missing Column Definitions

**Location:** `tables.limits_data.key_columns`

Add these new columns based on business-provided sample records:

### 2.1 `id1`
- **Definition:** "Additional identifier field"
- **Data Type:** INTEGER/VARCHAR
- **Usage:** "Internal reference. Not typically used in user queries."

### 2.2 `id2`
- **Definition:** "Limit identifier with class suffix (e.g., '300001-PRIMARY', '300002-SECONDARY')"
- **Data Type:** VARCHAR
- **Usage:** "Use pattern matching: WHERE id2 LIKE '%-PRIMARY' for primary limits. Contains limit_id + class suffix."

### 2.3 `grid_cc_id`
- **Definition:** "Grid currency code identifier"
- **Data Type:** INTEGER
- **Usage:** "Internal grid reference. Not typically used in queries."

### 2.4 `grid_cell_id`
- **Definition:** "Grid cell identifier for grid-based limits"
- **Data Type:** VARCHAR/DECIMAL
- **Usage:** "Used for grid-based aggregation. Example values: '0', '0.6704'"

### 2.5 `cur_id` ✅
- **Definition:** "Currency identifier code"
- **Data Type:** VARCHAR
- **Usage:** "Internal currency reference. Use 'curr' column for filtering by currency."
- **Note:** Use `cur_id` (confirmed - not `curr_id`)

### 2.6 `limit_usd` ❌ REMOVE
- **Status:** Ignore this field - do not add to data dictionary
- **Note:** Business confirmed to ignore `limit_usd`

### 2.7 `sec_id`
- **Definition:** "Security identifier"
- **Data Type:** VARCHAR/INTEGER
- **Usage:** "Reference to specific security. May be null for desk-level limits."

### 2.8 `sec_desc`
- **Definition:** "Security description"
- **Data Type:** VARCHAR
- **Usage:** "Security or limit description. Use for display and text search."

### 2.9 `issuer_id`
- **Definition:** "Issuer/counterparty identifier"
- **Data Type:** INTEGER/VARCHAR
- **Usage:** "Links to issuer_nm. Used for issuer-level filtering."

### 2.10 `issuer_nm` ✅
- **Definition:** "Issuer name"
- **Data Type:** VARCHAR
- **Usage:** "Human-readable issuer name. Used for issuer-level filtering and display."
- **Note:** Use `issuer_nm` (confirmed - not `issuer_mm`)

### 2.11 `ind_class_id`
- **Definition:** "Industry classification identifier"
- **Data Type:** INTEGER
- **Usage:** "Links to industry classification. May be null."

### 2.12 `ind_desc`
- **Definition:** "Industry description"
- **Data Type:** VARCHAR
- **Usage:** "Industry sector description. May be null."

### 2.13 `region_id`
- **Definition:** "Region identifier code"
- **Data Type:** INTEGER
- **Usage:** "Numeric region code. Use region_cd for filtering."

### 2.14 `wf_instance_id` (workflow instance) ✅
- **Definition:** "Workflow/approval instance identifier"
- **Data Type:** INTEGER
- **Usage:** "Links to approval workflows. Used with 'state' field for workflow tracking."
- **Note:** Use `wf_instance_id` (confirmed - not `wl_instance_id` or `inst_id`)

### 2.15 `st_dt` ✅
- **Definition:** "Start date for limit/extension period"
- **Data Type:** DATE
- **Usage:** "Start date for limit validity or extension period. Format: YYYY-MM-DD"
- **Note:** Use `st_dt` (confirmed - not `start_date`)

### 2.16 `limit_concatenate` ✅
- **Definition:** "Concatenated limit identifier string for display"
- **Data Type:** VARCHAR
- **Usage:** "Human-readable concatenated identifier (e.g., '3116802 / EQ VEGA /'). Use for display only."
- **Note:** Use `limit_concatenate` (confirmed - not `mtfi_concatenate`)

### 2.17 `pref_name`
- **Definition:** "Preferred/display name for limit"
- **Data Type:** VARCHAR
- **Usage:** "Short display name (e.g., 'Oil Prod Limit'). Use for reporting/display."

### 2.18 `IM_Column1`, `IM_Column2`, `IM_Column3`, `IM_Column4`
- **Definition:** "Additional metadata columns (Internal Management)"
- **Data Type:** VARCHAR/NUMERIC
- **Usage:** "Internal metadata fields. Not typically used in user queries."

---

## Section 3: Update Existing Column Definitions

**Location:** `tables.limits_data.key_columns`

### 3.1 Update `limit_class` definition
- **Current:** Lists "Decommissioned" as a value
- **Change:** Update values list to: `["ex-ZNDRY", "ex-PRIM", "Inner RAL", "Notice", "Outer RAL", "Primary", "Reporting", "Secondary"]`
- **Source:** Lines 130-139 in new file

### 3.2 Update `limit_group` definition
- **Current:** Simple examples like "PV01, RR & Gamma"
- **Change:** Update definition to mention letter-prefixed categories: "Many categories start with letters (A., B., C., etc.) like 'A. VaR & Stress Limits', 'C. PV01 Curve Risk Limits'. Total of 78 limit group categories."
- **Update usage:** Change example to: `WHERE limit_group LIKE '%PV01%' OR limit_group = 'C. PV01 Curve Risk Limits'`

### 3.3 Update `limit_type` definition
- **Current:** Basic examples like "PV01 Delta", "Gamma Vega"
- **Change:** Add note that limit_types are very specific and may be long strings like "CR01 / Bond / ExclNon-Corporates / ExclCcn Banks / A+ / IG Corp" or "Total Stress (w/ MSS) - MTM USD". Total of 122 limit type values.

### 3.4 Update `aggr_func_cd` definition
- **Current:** Generic description "Aggregation method: NET (allows offsets), GROSS (sum absolute), GRID, BY TRADE, etc."
- **Change:** Update to: "Aggregation method with 32 variants: GRID, GROSS, NET, GROSS BY SECURITY, SINGLE NAME, SINGLE INDUSTRY, NET LONG BY SEC, BY TRADE, etc. See data_samples for full list."

### 3.5 Update `meas_unit` definition
- **Current:** Limited examples "Currency/unit (USD, CAD, EUR, JPY, BBL, OZ, MT, MWH)"
- **Change:** Expand to include all 27 units: "Currency/unit with 27 variants: USD, CAD, EUR, JPY, AUD, BBL, BPS, BU, CHF, GBP, MT, MWH, MXN, NZD, OZ, Event, DAYS, etc. See data_samples for full list."
- **Note:** Use field name `meas_unit` (as shown in sample records), not `means_unit`

### 3.6 Update `state` definition ✅
- **Current (v2):** `["Active", "Pending Upload", "Extended", "InBreach", "Decommissioned"]`
- **Change:** Update to confirmed state values: `["Approved", "Pending", "Rejected"]`
- **Action:** Replace entire values list with: `["Approved", "Pending", "Rejected"]` (confirmed by business)

### 3.7 Update `extension` definition ✅
- **Current (v2):** "1 = temporary limit override active, 0 = normal"
- **Change:** Update to reflect confirmed logic: "Contains the value for the extended total limit when an extension is granted."
- **Action:** Update definition to: "Extension field will have a value for the extended total limit. When a limit extension is active, this field contains the extended limit amount."

### 3.8 Update critical_note for limit_group
- **Current:** Simple PV01 example
- **Change:** Update to: "⚠️ CRITICAL: Many limit_groups start with letters (A., B., C., etc.). When users mention 'PV01', match against patterns like 'C. PV01 Curve Risk Limits' or use pattern matching. Example: 'PV01 limits' → WHERE limit_group LIKE '%PV01%' OR limit_group = 'C. PV01 Curve Risk Limits'"

### 3.9 Update `letter_nm` definition
- **Current:** Basic examples
- **Change:** Update usage to note: "Use exact match for precision. For regional queries, can use ILIKE '%Canadian%' to match multiple Canadian desks. See business_glossary for regional patterns."

---

## Section 4: Update Business Glossary

**Location:** `business_glossary`

### 4.1 Update `market_risk_limits_overview.definition`
- **Current:** Basic definition
- **Change:** Update to match new metadata definition: "Enhanced market risk exposures system controlling exposure to market movements across all asset classes (rates, prices, FX, credit). Each limit has: (1) Risk metric (PV01, Delta, Gamma, Vega, VaR); (2) Risk measure (dollar amount), (3) Threshold (limit value). Utilization = exposure_amt / effective_limit. Breach = utilization >= 1.0."
- **Source:** Line 650 in new file

### 4.2 Update `key_metrics` in overview
- **Change:** Expand definitions to match new metadata:
  - **PV01:** "Interest rate sensitivity (dollar value of 1bp rate change) — includes Base PV01, PV01 by Curve"
  - **Delta:** "Price sensitivity change in value per $1 move in underlying — EQ DELTA, Oil Products Delta"
  - **Gamma:** "Delta sensitivity (convexity risk) — IR GAMMA, EQ GAMMA, 10 BPS IR GAMMA"
  - **Vega:** "Volatility sensitivity — IR VEGA, EQ VEGA, FX VEGA, commodity vegas"
  - **Add:** "Credit Limits": "single name and aggregated traded credit line limits"
  - **Add:** "Physical limits": "Physical commodity inventory exposures"
- **Source:** Lines 653-660 in new file

### 4.3 Update `limit_hierarchy`
- **Current:** "Trading Desk → Risk Category (limit_group) → Limit Type (limit_type) → Specific Limit"
- **Change:** "Trading Desk → Risk Category (limit_group) → Limit Type (limit_type) → Specific Limit → Workflow State"
- **Source:** Line 662 in new file

### 4.4 Update `business_terminology_mapping.risk_metrics`
- **Current:** Simple PV01 mapping with exact value "PV01"
- **Change:** Update critical_note to: "When user says 'PV01', use limit GROUP pattern matching like '%PV01%' (matches multiple PV01 categories like 'C. PV01 Curve Risk Limits', 'C. PV01 Curve Limits')"
- **Update exact_value:** Change to example like "C. PV01 Curve Risk Limits"
- **Source:** Lines 673-677 in new file

### 4.5 Add `regional_patterns` to terminology mapping
- **Change:** Add new section based on new metadata:
  ```json
  "regional_patterns": {
    "Canadian": ["Canadian Options", "Canadian Money Markets", "Canadian Government Bond Trading", "Canadian Prime Finance"],
    "US": ["US Delta One", "US Rates Trading", "US Treasury Trading", "Origination - US"],
    "European": ["European Prime Finance", "European Money Markets", "Europe Delta One"],
    "Asia": ["FX HK Trading", "China Fixed Income Trading", "China Funding"],
    "Global": ["Global Credit", "Global Equities", "Global Money Markets", "Global Prime Finance"]
  }
  ```
- **Source:** Lines 586-592 in new file

### 4.6 Add `user_aliases` section
- **Change:** Add new section:
  ```json
  "user_aliases": {
    "NET": ["net", "netted", "with offsets"],
    "GROSS": ["gross", "absolute", "no netting"],
    "SINGLE": ["single name", "single currency", "individual"]
  }
  ```
- **Source:** Lines 679-683 in new file

### 4.7 Update desk_names mapping
- **Current:** Simple example with "Canadian Options"
- **Change:** Add example from new metadata showing "Canadian Government Bond Trading" with pattern matching
- **Source:** Lines 667-671 in new file

---

## Section 5: Update Procedural Knowledge

**Location:** `procedural_knowledge`

### 5.1 Update `critical_field_mapping.rule`
- **Current:** Simple PV01 example
- **Change:** Update to: "User query 'PV01 limits' → WHERE limit_group LIKE '%PV01%' (matches 'C. PV01 Curve Risk Limits', 'C. PV01 Curve Limits', etc.) NOT limit_type"

### 5.2 Update `critical_field_mapping.examples`
- **Change:** Update all examples to use new limit_group format:
  - User query: "show me Primary PV01 limits"
  - Correct: "WHERE limit_group LIKE '%PV01%' AND limit_class = 'Primary'"
  - Incorrect: "WHERE limit_group = 'PV01' (no exact match - use pattern)"

### 5.3 Update query examples throughout
- **Change:** Update all SQL examples that reference limit_group to use pattern matching:
  - Change: `limit_group = 'PV01'` → `limit_group LIKE '%PV01%'` OR `limit_group = 'C. PV01 Curve Risk Limits'`
  - Change: `limit_group = 'Stress Limits'` → `limit_group LIKE '%Stress%'` OR `limit_group = 'A. VaR & Stress Limits'`
- **Impact:** Update examples in:
  - `minimal_unique_attributes.examples`
  - `common_query_patterns.patterns`
  - `trend_analysis.examples`
  - `critical_field_mapping.examples`
  - `query_reference.common_queries`

### 5.4 Update `aggr_func_cd` examples
- **Change:** Update examples to show new values like "SINGLE INDUSTRY", "Net UW Accepted Stress / UW Accepted", "Pending Upload"
- **Note:** "Pending Upload" appears as aggr_func_cd in sample - verify if this is correct or should be in state field

---

## Section 6: Important Notes & Considerations

### 6.1 Date Format
- **Keep:** YYYY-MM-DD format as specified (do not change)
- **Note:** Business samples show `date_mm` field with MM/DD/YYYY format, but this appears to be a separate display field. Main `date` field should remain YYYY-MM-DD.

### 6.2 Extension Handling
- **Update:** Extension logic - note that negative values in `effective_limit` can indicate limit increases
- **New metadata states:** "extension = 0 indicates active extensions (negative = increases)" - verify this logic matches actual usage

### 6.3 Column Name Priority
- **Rule:** Use field names from business-provided sample records as source of truth
- **Example:** Business shows `meas_unit` in samples (not `means_unit` from reference section)

---

## Section 7: Summary of Changes

### Reference Values Updates: 6 sections
1. `typical_letter_nm_values` - Replace with 129 desks
2. `typical_limit_groups` - Replace with 78 limit groups  
3. `typical_limit_classes` - Update to 8 values
4. Add `typical_limit_types` - Add examples (122 total)
5. Add `typical_aggr_func_cd` - Add full list (32 values)
6. Expand `meas_unit` examples - Add all 27 units

### New Columns to Add: FINAL Updated list
- ✅ **To Add:** id1, id2, grid_cc_id, grid_cell_id, cur_id, curr, sec_id, sec_desc, issuer_id, issuer_nm, ind_class_id, industry, region_id, wf_instance_id, st_dt, limit_concatenate, pref_name, LM_Column1-4
- ✅ **Core Fields (Already exist but verify):** exposure_amt, Original_limit, effective_limit, utilization, rating_ceiling
- ❌ **Do NOT Add:** limit_usd (ignore), ind_desc (use industry instead), issuer_mm (use issuer_nm instead)

### Updated Column Definitions: 9 columns
- limit_class, limit_group, limit_type, aggr_func_cd, meas_unit, state, extension, letter_nm, critical_note

### Business Glossary Updates: 7 sections
- Overview definition, key metrics, limit hierarchy, risk_metrics mapping, add regional_patterns, add user_aliases, update desk_names

### Procedural Knowledge Updates: 4+ sections
- Critical field mapping, query examples, aggregation examples, pattern matching updates

**Total Impact:** ~40+ specific changes across the JSON file

---

## Next Steps

1. Review contradictions highlighted below
2. Use this document as a checklist when creating `data_dictionary_risk_v3.json`
3. Verify any discrepancies with business users before finalizing

---

# CONTRADICTIONS TO REVIEW ⚠️

**IMPORTANT:** These contradictions need to be resolved with business users before creating v3. Use business-provided details as source of truth, but verify these discrepancies.

---

## 🔴 CRITICAL CONTRADICTIONS

### Contradiction 1: Date Field Names and Formats ✅ RESOLVED
- **CSV Header:** Shows single `date` column (format: YYYY-MM-DD, e.g., "2024-10-09")
- **Business Sample Records:** Show `date_mm` field (format: MM/DD/YYYY, e.g., "12/2/2025")
- **Business Metadata Rule (line 611):** Says "Date format is MM/DD/YYYY, not YYYY-MM-DD"
- **Contradiction:** CSV uses YYYY-MM-DD format, business metadata says MM/DD/YYYY
- **✅ RESOLUTION:** 
  - **Keep YYYY-MM-DD format** (e.g., "2024-10-09")
  - Only `date` field exists (ignore `date_mm` from business samples - it's not a real field)
  - Format: YYYY-MM-DD

### Contradiction 2: Workflow/Instance ID Field Names ✅ RESOLVED
- **CSV Header:** Shows `wl_instance_id` (column 31)
- **Business Metadata (line 644):** References `wf_instance_id` (says "wf_instance_id links to approval workflows")
- **Business Sample Records:** Show `inst_instance_id` (with date values like "12/10/2025")
- **Contradiction:** Three different field names for what seems like the same concept
- **✅ RESOLUTION:** 
  - **Use `wf_instance_id`** as the field name
  - This links to approval workflows per business metadata

### Contradiction 3: Extension Field Logic ✅ RESOLVED
- **CSV Data:** Shows `extension` as 0 or 1
- **Business Metadata (line 610):** Says "extension = 0 indicates active extensions (negative = increases)"
- **Business Sample Records:** Show `extension` as 0 or 1, BUT `effective_limit` can be negative (e.g., "-40000000")
- **Contradiction:** Logic unclear - what do values mean?
- **✅ RESOLUTION:** 
  - **`extension` field will have a value for the extended total limit**
  - This means when an extension is granted, the `extension` field contains the extended limit amount
  - Update definition to reflect this: "Contains the extended total limit value when a limit extension is active"

### Contradiction 4: State Field Values ✅ RESOLVED
- **CSV Data:** Shows values like "Active", "Pending Upload"
- **Business Metadata (line 615):** Lists states as `["Approved", "Pending", "Rejected", "In Review"]`
- **Business Sample Records:** Show state values but samples don't explicitly show state field
- **V2 Current Definition:** Lists `["Active", "Pending Upload", "Extended", "InBreach", "Decommissioned"]`
- **Contradiction:** Three completely different sets of state values
- **✅ RESOLUTION:** 
  - **State values:** `["Approved", "Pending", "Rejected"]`
  - Update state field definition to use these three values

---

## 🟡 MEDIUM PRIORITY CONTRADICTIONS

### Contradiction 5: Field Name - meas_unit vs means_unit ✅ RESOLVED
- **CSV Header:** Uses `meas_unit` (column 10)
- **Business Reference Section:** Uses `means_unit` (line 375 - typo)
- **Business Sample Records:** Use `meas_unit` (lines 415, 450, 485, 520, 555)
- **Contradiction:** Reference section has typo "means_unit" but samples use "meas_unit"
- **✅ RESOLUTION:** **Use `meas_unit`** (confirmed - as shown in samples)

### Contradiction 6: Column Name - limit_concatenate vs mtfi_concatenate ✅ RESOLVED
- **CSV Header:** Shows `limit_concatenate` (column 36)
- **Business Sample Records:** Show `mtfi_concatenate` (lines 435, 470, 505, 540, 575)
- **Contradiction:** Different field names for what appears to be the same field
- **✅ RESOLUTION:** **Use `limit_concatenate`** (from CSV header - confirmed)

### Contradiction 7: Start Date Field Name ✅ RESOLVED
- **CSV Header:** Shows `st_dt` (column 34)
- **Business Sample Records:** Show `start_date` (lines 433, 468, 503, 538, 573)
- **Contradiction:** Different field names
- **✅ RESOLUTION:** **Use `st_dt`** (from CSV header - confirmed)

### Contradiction 8: Aggregation Function Value ✅ RESOLVED
- **CSV Data:** aggr_func_cd shows values like "NET", "GROSS", "SINGLE NAME", "BY TRADE"
- **Business Reference List:** 32 valid values (lines 342-374)
- **Business Sample Record (line 521):** Shows `"aggr_func_cd": "Pending Upload"`
- **Contradiction:** "Pending Upload" appears as aggr_func_cd but it's not in the reference list
- **✅ RESOLUTION:** 
  - **"Pending Upload" is a DATA QUALITY ISSUE** - not a valid aggregation function code
  - Do not include "Pending Upload" as a valid value in the data dictionary
  - This is an exception/data quality issue in the data

### Contradiction 9: Limit Group Values Format - CSV vs Business Reference ✅ RESOLVED
- **CSV Data:** Shows values like:
  - "Outer RAL"
  - "RR & Gamma" 
  - "Asset"
  - "Stress Limits"
  - "PV01"
- **Business Reference:** Lists 78 values, many with letter prefixes:
  - "A. VaR & Stress Limits"
  - "C. PV01 Curve Risk Limits"
  - "L. Single Name Traded Credit Limits"
- **Business Sample Records:** Show letter-prefixed format (e.g., "Z Reporting Only", "A. VaR & Stress Limits")
- **Contradiction:** CSV shows simpler names without letter prefixes, business reference shows letter-prefixed names
- **✅ RESOLUTION:** **Use business reference format** (letter-prefixed) - confirmed

---

## 🟢 MINOR CONTRADICTIONS / FIELD PRESENCE

### Contradiction 10: Additional Fields in Business Samples Not in CSV ✅ RESOLVED
- **CSV Header:** 41 columns total
- **Business Sample Records:** Show additional fields:
  - `date_mm` - Not in CSV header
  - `inst_id` - Not in CSV header  
  - `inst_instance_id` - Not in CSV header
- **✅ RESOLUTION:** 
  - Only `date` field exists (not `date_mm`)
  - Ignore `date_mm`, `inst_id`, `inst_instance_id` from business samples - these are not actual fields

### Contradiction 11: Field Name Variations ✅ RESOLVED
- **CSV Has:** `curr` (currency code) and `curr_id` (currency identifier)
- **Business Samples Show:** `cur_id` (line 423) - different spelling
- **Contradiction:** CSV shows `curr_id`, business shows `cur_id`
- **✅ RESOLUTION:** **Use `cur_id`** (as shown in business samples - confirmed)

---

## 📊 COLUMN-BY-COLUMN COMPARISON: CSV vs Business Details

### Complete Field Mapping

| CSV Header (41 cols) | Business Samples | Status | Notes |
|---------------------|------------------|--------|-------|
| `date` | `date` | ✅ Match | CSV: YYYY-MM-DD, Business also has `date_mm` (MM/DD/YYYY) |
| `letter_nm` | `letter_nm` | ✅ Match | |
| `limit_id` | `limit_id` | ✅ Match | |
| `id1` | `id1` | ✅ Match | |
| `id2` | `id2` | ✅ Match | |
| `limit_class` | `limit_class` | ✅ Match | |
| `limit_group` | `limit_group` | ✅ Match | Format differs: CSV simple, Business letter-prefixed |
| `limit_type` | `limit_type` | ✅ Match | |
| `limit_desc` | `limit_desc` | ✅ Match | |
| `meas_unit` | `meas_unit` | ✅ Match | Business ref section has typo `means_unit` |
| `aggr_func_cd` | `aggr_func_cd` | ✅ Resolved | "Pending Upload" is data quality issue - not valid value |
| `rating_floor` | `rating_floor` | ✅ Match | |
| `rating_ceiling` | `rating_ceiling` | ✅ Include | Rating ceiling - now included |
| `unofficial_flag` | `unofficial_flag` | ✅ Match | |
| `exposure_amt` | `exposure_amt` | ✅ Match | |
| `Original_limit` | ❌ Missing | ✅ Include | Core field - must be in v3 (from CSV) |
| `effective_limit` | `effective_limit` | ✅ Match | Core field |
| `utilization` | ❌ Missing | ✅ Include | Core field - must be in v3 (from CSV) |
| `grid_cc_id` | ❌ Missing | ✅ Include | Grid currency code identifier - now included |
| `grid_cell_id` | `grid_cell_id` | ✅ Match | |
| `curr_id` | `cur_id` | ✅ Use `cur_id` | Use `cur_id` spelling (confirmed) |
| `curr` | ❌ Missing | ✅ Include | Currency code - now included |
| `sec_id` | `sec_id` | ✅ Match | |
| `sec_desc` | `sec_desc` | ✅ Match | |
| `issuer_id` | `issuer_id` | ✅ Match | |
| `issuer_nm` | `issuer_nm` | ✅ Match | Use `issuer_nm` (confirmed - not `issuer_mm`) |
| `ind_class_id` | `ind_class_id` | ✅ Match | |
| `industry` | ❌ Missing | ✅ Include | Industry name - use `industry` not `ind_desc` |
| `ind_desc` | ❌ Missing | ❌ Use `industry` | Use `industry` field name instead (they are the same) |
| `region_id` | `region_id` | ✅ Match | |
| `region_cd` | `region_cd` | ✅ Match | |
| `wl_instance_id` | ❌ Missing | ✅ Use `wf_instance_id` | Use `wf_instance_id` per business confirmation |
| `state` | `state` | ✅ Resolved | Values: ["Approved", "Pending", "Rejected"] |
| `extension` | `extension` | ✅ Resolved | Contains value for extended total limit |
| `st_dt` | `st_dt` | ✅ Match | Use `st_dt` (confirmed) |
| `end_dt` | `end_dt` | ✅ Match | |
| `limit_concatenate` | `limit_concatenate` | ✅ Match | Use `limit_concatenate` (confirmed) |
| `pref_name` | `pref_name` | ✅ Match | |
| `LM_Column1-4` | `IM_Column1-4` | ⚠️ Different | Name prefix: CSV `LM_` vs Business `IM_` |

### Fields ONLY in Business Samples (Not in CSV - IGNORE):
1. **`date_mm`** - ❌ Not a real field (only `date` exists)
2. **`inst_id`** - ❌ Not a real field (use `wf_instance_id` instead)
3. **`inst_instance_id`** - ❌ Not a real field (use `wf_instance_id` instead)
4. **`limit_usd`** - ❌ Ignore - do not add (confirmed)
5. **`mtfi_concatenate`** - ❌ Not a real field (use `limit_concatenate` instead)

### Fields ONLY in CSV - FINAL Status:
1. **`Original_limit`** - ✅ **INCLUDE** (core field - must be in v3)
2. **`utilization`** - ✅ **INCLUDE** (core field - must be in v3)
3. **`exposure_amt`** - ✅ **INCLUDE** (core field - already exists, verify definition)
4. **`effective_limit`** - ✅ **INCLUDE** (core field - already exists, verify definition)
5. **`rating_ceiling`** - ✅ **INCLUDE** (rating ceiling - now included)
6. **`grid_cc_id`** - ✅ **INCLUDE** (grid currency code identifier - now included)
7. **`curr`** - ✅ **INCLUDE** (currency code - now included)
8. **`industry`** - ✅ **INCLUDE** (industry name - use `industry` not `ind_desc`)
9. **`wl_instance_id`** - ✅ Use `wf_instance_id` instead (confirmed)

---

## 📋 SUMMARY OF CONTRADICTIONS - ALL RESOLVED ✅

### ✅ Resolved Field Names:
1. **Date field:** Only `date` exists (YYYY-MM-DD format, e.g., "2024-10-09") - ignore `date_mm`
2. **Workflow ID:** Use `wf_instance_id` (links to approval workflows)
3. **Extension field:** Contains value for the extended total limit
4. **State values:** Use `["Approved", "Pending", "Rejected"]`
5. **Field names:** 
   - ✅ `meas_unit` (not `means_unit`)
   - ✅ `limit_concatenate` (not `mtfi_concatenate`)
   - ✅ `st_dt` (not `start_date`)
   - ✅ `cur_id` (not `curr_id`)

### ✅ Resolved Formats:
1. **Limit group format:** Use letter-prefixed format from business reference (e.g., "A. VaR & Stress Limits", "C. PV01 Curve Risk Limits")

### ✅ All Questions Resolved:
1. ✅ **aggr_func_cd:** "Pending Upload" is a **data quality issue** - not a valid value

---

## 🎯 RESOLUTION SUMMARY & REMAINING QUESTIONS

### ✅ All Major Contradictions Resolved:
1. ✅ **Date format:** YYYY-MM-DD, only `date` field exists
2. ✅ **Workflow ID:** Use `wf_instance_id`
3. ✅ **Extension field:** Contains value for extended total limit
4. ✅ **State values:** ["Approved", "Pending", "Rejected"]
5. ✅ **Field names:** `meas_unit`, `limit_concatenate`, `st_dt`, `cur_id`
6. ✅ **Limit group format:** Letter-prefixed from business reference

### ❓ Remaining Questions - ALL RESOLVED ✅:

1. **aggr_func_cd value:** ✅ RESOLVED
   - Business sample record shows `"aggr_func_cd": "Pending Upload"` (line 521 in new file)
   - **Resolution:** This is a **data quality issue** - "Pending Upload" is NOT a valid aggregation function code
   - **Action:** Do not include "Pending Upload" as a valid value. Document as data quality exception if needed.

2. **issuer_nm vs issuer_mm:** ✅ RESOLVED
   - CSV header shows `issuer_nm` 
   - Business sample records show `issuer_mm`
   - **Resolution:** Use **`issuer_nm`** (from CSV header - confirmed)

3. **Fields only in CSV (not in business samples):** ✅ RESOLVED
   - **Core fields to include:** `exposure_amt`, `Original_limit`, `effective_limit`, `utilization` - these should be in the data dictionary
   - **Fields to exclude:** `rating_ceiling`, `grid_cc_id`, `curr`, `industry` - do not include these
   - **Action:** Ensure `exposure_amt`, `Original_limit`, `effective_limit`, `utilization` are properly defined in v3

4. **limit_usd field:** ✅ RESOLVED
   - Appears in business sample records but not in CSV header
   - **Resolution:** **Ignore `limit_usd`** - do not add to data dictionary

### For v3 Creation:
- ✅ Use resolved field names and formats
- ✅ Update state values to ["Approved", "Pending", "Rejected"]
- ✅ Use letter-prefixed limit_group format
- ✅ Use `wf_instance_id` for workflow tracking
- ✅ Update extension field definition to reflect it contains extended total limit value
- ✅ Ensure core fields are included: `exposure_amt`, `Original_limit`, `effective_limit`, `utilization`
- ✅ Use `issuer_nm` (not `issuer_mm`)
- ✅ Include: `rating_ceiling`, `grid_cc_id`, `curr`, `industry`
- ✅ Do NOT include: `limit_usd`, `ind_desc` (use `industry` instead)
- ✅ Note: "Pending Upload" in aggr_func_cd is a data quality issue - not a valid value
- ✅ Final column count: **41 columns**

---

## 📋 FINAL COLUMN LIST FOR V3 (41 COLUMNS)

### Complete List - Ready for v3 JSON Creation

1. `date`
2. `letter_nm`
3. `limit_id`
4. `id1`
5. `id2`
6. `limit_class`
7. `limit_group`
8. `limit_type`
9. `limit_desc`
10. `meas_unit`
11. `aggr_func_cd`
12. `rating_floor`
13. `rating_ceiling` ✅ (included)
14. `unofficial_flag`
15. `exposure_amt`
16. `Original_limit`
17. `effective_limit`
18. `utilization`
19. `grid_cc_id` ✅ (included)
20. `grid_cell_id`
21. `cur_id` (was `curr_id` in CSV)
22. `curr` ✅ (included)
23. `sec_id`
24. `sec_desc`
25. `issuer_id`
26. `issuer_nm`
27. `ind_class_id`
28. `industry` ✅ (use `industry` not `ind_desc`)
29. `region_id`
30. `region_cd`
31. `wf_instance_id` (was `wl_instance_id` in CSV)
32. `state`
33. `extension`
34. `st_dt`
35. `end_dt`
36. `limit_concatenate`
37. `pref_name`
38. `LM_Column1`
39. `LM_Column2`
40. `LM_Column3`
41. `LM_Column4`

**Total: 41 columns**

---

*Document Created: Based on comparison of data_dictionary_risk_v2.json and new_limits_data_details.json*  
*Final Review Completed: All 41 columns confirmed - Ready for v3 JSON creation*

