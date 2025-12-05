# Developer Setup Instructions: Data Dictionary

## Critical Rule: Table Name = CSV Filename
- **CSV file:** `limits_data.csv` → **Table name:** `limits_data`
- **JSON must have:** `"tables": { "limits_data": {...} }`
- **They must match exactly** (case-sensitive)

## Setup Steps

### 1. Ensure Data Dictionary File Exists
- **Location:** `external/config/data_dictionary/data_dictionary_risk.json`
- Verify it exists and is valid JSON
- **Note:** This file should contain the approved v3 structure with all 41 columns

### 2. Ensure CSV File Exists
- **Path:** `data/duckdb/limits_data.csv` (or `data/duckdb/agent_data/MR/limits_data.csv`)
- **Filename must be:** `limits_data.csv` (matches table name in JSON)
- **Default data directory:** `data/duckdb` (configured in `external/config/agent/queryagent_planner.json`)

## File Structure
```
project/
├── external/config/data_dictionary/
│   └── data_dictionary_risk.json  ← Data dictionary file
├── external/config/data_analysis_agents/
│   ├── risk_agent.json  ← Points to data_dictionary_risk.json (default)
│   └── mr_limits_analysis.json  ← Points to data_dictionary_risk.json (default)
└── data/duckdb/
    └── limits_data.csv  ← CSV file (filename = table name)
```

## Verification Checklist
- [ ] JSON file exists: `external/config/data_dictionary/data_dictionary_risk.json`
- [ ] CSV file exists: `data/duckdb/limits_data.csv`
- [ ] Table name matches: JSON has `"tables": { "limits_data": {...} }` and CSV is `limits_data.csv`

## No Code Changes Required
- Code defaults to `data_dictionary_risk.json` if no custom file specified
- Code loads from: `external/config/data_dictionary/{filename}`
- Config files use default name (no updates needed)

## Where the Config is Used
- `invoke_node()` — loads tables, business_context, key_columns from the dictionary
- `end_node()` — loads procedural_knowledge from the dictionary
- `nl_to_sql_planner.py` — loads procedural_knowledge for SQL generation
- SocketIO/API routes — pass the `data_dict_file` from agent config to the agent

## Notes
- Table name in JSON must match CSV filename (without `.csv` extension)
- All paths are relative to project root
- Default data directory is `data/duckdb` (configurable in `queryagent_planner.json`)
- Code automatically uses `data_dictionary_risk.json` as default (no config changes needed)
