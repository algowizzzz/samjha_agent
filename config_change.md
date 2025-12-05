# Developer Setup Instructions: v3 Data Dictionary

## Critical Rule: Table Name = CSV Filename
- **CSV file:** `limits_data.csv` → **Table name:** `limits_data`
- **JSON must have:** `"tables": { "limits_data": {...} }`
- **They must match exactly** (case-sensitive)

## Setup Steps

### 1. Copy v3 JSON File
- **Location:** `external/config/data_dictionary/data_dictionary_risk_v3.json`
- Verify it exists and is valid JSON

### 2. Update Agent Config Files
- **File:** `external/config/data_analysis_agents/risk_agent.json`
  - Change: `"data_dict_file": "data_dictionary_risk.json"` → `"data_dict_file": "data_dictionary_risk_v3.json"`
- **File:** `external/config/data_analysis_agents/mr_limits_analysis.json`
  - Change: `"data_dict_file": "data_dictionary_risk.json"` → `"data_dict_file": "data_dictionary_risk_v3.json"`

### 3. Ensure CSV File Exists
- **Path:** `data/duckdb/limits_data.csv` (or `data/duckdb/agent_data/MR/limits_data.csv`)
- **Filename must be:** `limits_data.csv` (matches table name in JSON)
- **Default data directory:** `data/duckdb` (configured in `external/config/agent/queryagent_planner.json`)

## File Structure
```
project/
├── external/config/data_dictionary/
│   └── data_dictionary_risk_v3.json  ← v3 JSON file
├── external/config/data_analysis_agents/
│   ├── risk_agent.json  ← Updated to point to v3
│   └── mr_limits_analysis.json  ← Updated to point to v3
└── data/duckdb/
    └── limits_data.csv  ← CSV file (filename = table name)
```

## Verification Checklist
- [ ] JSON file exists: `external/config/data_dictionary/data_dictionary_risk_v3.json`
- [ ] Config files updated: Both agent configs point to `data_dictionary_risk_v3.json`
- [ ] CSV file exists: `data/duckdb/limits_data.csv`
- [ ] Table name matches: JSON has `"tables": { "limits_data": {...} }` and CSV is `limits_data.csv`

## No Code Changes Required
- Code already supports this via `state['config_files']['data_dict_file']`
- Code loads from: `external/config/data_dictionary/{filename}`
- Falls back to default if not specified

## Where the Config is Used
- `invoke_node()` — loads tables, business_context, key_columns from the dictionary
- `end_node()` — loads procedural_knowledge from the dictionary
- `nl_to_sql_planner.py` — loads procedural_knowledge for SQL generation
- SocketIO/API routes — pass the `data_dict_file` from agent config to the agent

## Notes
- Table name in JSON must match CSV filename (without `.csv` extension)
- All paths are relative to project root
- Default data directory is `data/duckdb` (configurable in `queryagent_planner.json`)
