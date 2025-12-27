# Input Data/Config Files Reference

Complete list of all non-code input data and configuration files used by the Parquet Agent.

## 📋 CATEGORY 1: PROMPTS

**Location:** `external/config/prompts/`

### Files:
- **`decider.md`**
  - **Purpose:** Main Decider prompt template
  - **Used by:** `external/agent/decider.py` (`load_decider_prompt()`)
  - **Contains:** 
    - Step-by-step instructions for LLM to create QuerySpec
    - Query type detection (NEW_QUERY, FOLLOW_UP, USER_ANSWER)
    - Investigation-first approach
    - No-time-by-default policy
    - Schema definitions and examples
  - **Format:** Markdown with JSON schema examples

---

## 📋 CATEGORY 2: DOMAIN CONFIGURATION

**Location:** `external/config/domains/`

### Files:
- **`ecomm_domain.md`**
  - **Purpose:** E-commerce domain rules, entities, metrics, time rules
  - **Used by:** 
    - `external/agent/parquet_agent.py` (`load_domain_md()`)
    - `external/tools/parquet_agent/search_glossary.py`
    - `external/agent/sql_gate.py` (for domain-specific rules)
  - **Contains:**
    - Core entities (sales, customers, products, inventory)
    - Metric dictionary (revenue, quantity, count, etc.)
    - Time column defaults (order_date, signup_date)
    - Grain examples (one row per region, product, etc.)
    - Listing rules (empty metrics allowed)
    - Guardrails and constraints
  - **Format:** Markdown with YAML-like structured sections

- **`mr_domain.md`** (if exists)
  - **Purpose:** Market Risk domain configuration
  - **Used by:** Same as above
  - **Auto-selection:** Triggered when query contains 'market risk', 'limits', or 'mr'

### Domain Selection Logic:
- **Default:** `ecomm`
- **Auto-switch to `mr`** if query contains: 'market risk', 'limits', or 'mr'
- **File pattern:** `{domain}_domain.md`
- **Location:** `external/config/domains/{domain}_domain.md`

---

## 📋 CATEGORY 3: TOOL CONFIGURATIONS

**Location:** `external/config/tools/`

### Parquet Agent Tools:
1. **`parquet_agent.json`**
   - Main agent tool configuration
   - Used by: `external/agent/base_agent.py`

2. **`list_dir.json`**
   - Directory listing tool
   - **Config fields:**
     - `data_directory`: `"external/datawarehouse"` (default)
   - Used by: `external/tools/parquet_agent/list_dir.py`

3. **`inspect_table.json`**
   - Schema inspection tool
   - **Config fields:**
     - `data_directory`: `"external/datawarehouse"` (default)
   - Used by: `external/tools/parquet_agent/inspect_table.py`

4. **`preview_rows.json`**
   - Row preview tool
   - **Config fields:**
     - `data_directory`: `"external/datawarehouse"` (default)
   - Used by: `external/tools/parquet_agent/preview_rows.py`

5. **`execute_sql.json`**
   - SQL execution tool
   - **Config fields:**
     - `data_directory`: `"external/datawarehouse"` (default)
   - Used by: `external/tools/parquet_agent/execute_sql.py`

6. **`search_glossary.json`**
   - Glossary search tool
   - **Config fields:**
     - `domain_directory`: `"external/config/domains"` (default)
   - Used by: `external/tools/parquet_agent/search_glossary.py`

7. **`nl_to_sql_planner.json`**
   - SQL generation tool
   - Used by: `external/tools/parquet_agent/nl_to_sql_planner.py`

8. **`sql_plan_updater.json`**
   - SQL plan update tool
   - Used by: `external/tools/parquet_agent/sql_plan_updater.py`

9. **`query_result_evaluator.json`**
   - Result evaluation tool
   - Used by: `external/tools/parquet_agent/query_result_evaluator.py`

10. **`query_safety_validator.json`**
    - Safety validation tool
    - Used by: `external/tools/parquet_agent/query_safety_validator.py`

### Tavily Tools (External):
11. **`tavily_web_search.json`**
12. **`tavily_news_search.json`**
13. **`tavily_research_search.json`**
14. **`tavily_domain_search.json`**

### Tool Config Structure:
Each tool JSON contains:
- `name`: Tool identifier
- `implementation`: Python class path
- `description`: Tool description
- `version`: Version number
- `enabled`: Boolean flag
- `inputSchema`: JSON schema for input
- `outputSchema`: JSON schema for output
- `metadata`: Author, category, tags
- **Tool-specific configs:**
  - `data_directory`: For data access tools (default: `"external/datawarehouse"`)
  - `domain_directory`: For glossary tool (default: `"external/config/domains"`)

---

## 📋 CATEGORY 4: DATA FILES

**Location:** `external/datawarehouse/`

### Structure:
```
external/datawarehouse/
├── ECommerce/
│   ├── sample_sales_data.csv
│   ├── sample_customer_data.csv
│   └── sample_inventory_data.csv
└── MR/
    └── limits_data.csv
```

### Usage:
- **Read by:** Tools via `data_directory` config (default: `external/datawarehouse`)
- **Tools that use data:**
  - `list_dir`: Lists files in domain folders
  - `inspect_table`: Inspects schema of CSV files
  - `preview_rows`: Previews sample rows
  - `execute_sql`: Executes SQL queries against CSV files (via DuckDB)

### Path Resolution:
- Tools normalize paths: `ecomm/` → `ECommerce/`
- View names extracted from file stems: `sample_sales_data.csv` → `sample_sales_data`
- SQL uses view names, not full paths

---

## 📋 CATEGORY 5: JSON SCHEMAS

**Location:** `external/schemas/`

### Files:
1. **`query_spec.schema.json`**
   - Query specification structure
   - Defines: business_question, output_shape, start_table, grain, time, metrics, dimensions, filters, joins
   - Used by: Schema validators to validate Decider output

2. **`query_spec_status.schema.json`**
   - Status tracking structure
   - Defines: status fields (missing, inferred, verified, conflict, defaulted)
   - Used by: SQL gate to check if QuerySpec is ready

3. **`decider_output.schema.json`**
   - Decider output structure
   - Defines: action, query_type, decisions, query_spec, query_spec_status, investigation_plan
   - Used by: `external/agent/decider.py` validation

4. **`executor_report.schema.json`**
   - Executor report structure
   - Defines: SUCCESS, ERROR, ASK_USER status branches
   - Used by: `external/agent/executor_nodes.py` validation

5. **`investigation_plan_step.schema.json`**
   - Investigation step structure
   - Defines: tool_name, args, fills_gap, success_condition
   - Used by: Investigation plan validation

6. **`policy_limits.schema.json`**
   - Policy limits structure
   - Defines: max_rows, max_execution_time, etc.
   - Used by: Policy enforcement

### Usage:
- **Validated by:** `external/agent/schema_validators.py`
- **Purpose:** Ensure LLM output matches expected structure
- **Format:** JSON Schema Draft 2020-12

---

## 📋 CATEGORY 6: OTHER CONFIG

**Location:** `external/config/agent/`

### Files:
- **`queryagent_planner.json`**
  - Query agent planner configuration (if used)
  - Purpose: Additional planner configuration

---

## 🔧 CONFIGURATION POINTS

### 1. Domain Selection Logic
- **Default:** `ecomm`
- **Auto-switch to `mr`** if query contains: 'market risk', 'limits', or 'mr'
- **File pattern:** `{domain}_domain.md`
- **Location:** `external/config/domains/{domain}_domain.md`
- **Code:** `external/agent/parquet_agent.py::load_domain_md()`

### 2. Data Directory
- **Default:** `external/datawarehouse`
- **Configurable per tool** via tool JSON config (`data_directory` field)
- **Tools using it:**
  - `list_dir`
  - `inspect_table`
  - `preview_rows`
  - `execute_sql`
- **Code:** Tool implementations read from `self.config.get("data_directory", "external/datawarehouse")`

### 3. Domain Directory
- **Default:** `external/config/domains`
- **Used by:** `search_glossary` tool
- **Configurable via:** Tool JSON config (`domain_directory` field)
- **Code:** `external/tools/parquet_agent/search_glossary.py`

### 4. Prompt Loading
- **Location:** `external/config/prompts/decider.md`
- **Code:** `external/agent/decider.py::load_decider_prompt()`
- **Format:** Markdown with embedded JSON schema examples

---

## 📝 SUMMARY

| Category | Count | Location | Key Files |
|----------|-------|----------|-----------|
| **Prompts** | 1 | `external/config/prompts/` | `decider.md` |
| **Domains** | 1+ | `external/config/domains/` | `ecomm_domain.md`, `mr_domain.md` |
| **Tool Configs** | 14 | `external/config/tools/` | `parquet_agent.json`, `list_dir.json`, etc. |
| **Data Files** | 4 | `external/datawarehouse/` | CSV files in domain folders |
| **Schemas** | 6 | `external/schemas/` | `query_spec.schema.json`, etc. |
| **Other Config** | 1 | `external/config/agent/` | `queryagent_planner.json` |

**Total:** ~27 files (excluding data CSV files)

---

## 🔍 HOW TO MODIFY

1. **Change Decider behavior:** Edit `external/config/prompts/decider.md`
2. **Add new domain:** Create `external/config/domains/{domain}_domain.md`
3. **Change data location:** Update `data_directory` in tool JSON configs
4. **Add new tool:** Create `external/config/tools/{tool_name}.json`
5. **Modify schemas:** Edit `external/schemas/*.schema.json` (requires code updates)

---

**Last Updated:** 2025-12-27

