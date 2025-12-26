# MCP Tools Implementation Guide

> **Target**: Parquet Agent Executor Tools  
> **Data Source**: `mock_datawarehouse/`  
> **Domain Config**: `domain_instructions/`

---

## 1. Overview

The Executor in the Decider/Executor architecture requires **9 MCP tools**. Some exist and need adaptation; others must be built from scratch.

### Tool Summary

| Tool | Status | Action Required |
|------|--------|-----------------|
| `list_dir` | Adapt | Wrap `duckdb_list_files` |
| `inspect_table` | Adapt | Wrap `duckdb_describe_table` |
| `preview_rows` | **NEW** | Create |
| `search_glossary` | **NEW** | Create |
| `nl_to_sql_planner` | Rewrite | Change signature to spec |
| `sql_plan_updater` | **NEW** | Create (LLM-based) |
| `query_safety_validator` | Adapt | Align schema to spec |
| `execute_sql` | Adapt | Wrap `duckdb_query` |
| `query_result_evaluator` | Adapt | Align schema to spec |

---

## 2. Data Sources

### 2.1 Mock Data Warehouse Structure

```
mock_datawarehouse/
├── ECommerce/
│   ├── sample_customer_data.csv    # Customer master
│   ├── sample_inventory_data.csv   # Product inventory
│   └── sample_sales_data.csv       # Order transactions
└── MR/
    └── limits_data.csv             # Market risk limits
```

### 2.2 ECommerce Domain Schemas

**`sample_customer_data.csv`**
| Column | Type | Description |
|--------|------|-------------|
| customer_id | STRING | Primary key (C001, C002...) |
| customer_name | STRING | Full name |
| email | STRING | Email address |
| country | STRING | Country code |
| signup_date | DATE | Registration date |
| total_purchases | DECIMAL | Lifetime purchase amount |
| customer_tier | STRING | Gold/Silver/Bronze |

**`sample_inventory_data.csv`**
| Column | Type | Description |
|--------|------|-------------|
| product_id | STRING | Primary key (P001, P002...) |
| product_name | STRING | Product name |
| category | STRING | Product category |
| stock_quantity | INTEGER | Current stock |
| reorder_level | INTEGER | Reorder threshold |
| unit_cost | DECIMAL | Cost price |
| unit_price | DECIMAL | Selling price |
| supplier | STRING | Supplier name |

**`sample_sales_data.csv`**
| Column | Type | Description |
|--------|------|-------------|
| order_id | INTEGER | Primary key |
| customer_id | STRING | FK to customers |
| product | STRING | Product name |
| category | STRING | Product category |
| quantity | INTEGER | Units ordered |
| price | DECIMAL | Unit price |
| order_date | DATE | Transaction date |
| region | STRING | Sales region |

### 2.3 Domain Instructions Location

```
domain_instructions/
└── ecomm_domain.md    # ECommerce domain rules
```

---

## 3. Tool Implementations

### 3.1 `list_dir` (Adapt from `duckdb_list_files`)

**Purpose**: Discover datasets in a domain folder.

**Input Schema** (spec-compliant):
```json
{
  "args": {
    "type": "object",
    "required": ["path"],
    "properties": {
      "path": { "type": "string", "description": "Domain folder path (e.g., 'ECommerce')" }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["entries"],
    "properties": {
      "entries": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["name", "path", "type"],
          "properties": {
            "name": { "type": "string" },
            "path": { "type": "string" },
            "type": { "enum": ["file", "dir"] }
          }
        }
      }
    }
  }
}
```

**Implementation**:
```python
# File: external/tools/parquet_agent/list_dir.py

from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool

class ListDirTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "list_dir",
            "description": "List files and directories in a domain folder",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.base_path = Path(self.config.get("data_directory", "mock_datawarehouse"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "Domain folder path"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["entries"],
            "properties": {
                "entries": {"type": "array", "items": {"type": "object"}}
            }
        }

    def execute(self, arguments):
        path = arguments["path"]
        target = self.base_path / path
        
        if not target.exists():
            return {"entries": [], "error": f"Path not found: {path}"}
        
        entries = []
        for item in target.iterdir():
            entries.append({
                "name": item.name,
                "path": str(item.relative_to(self.base_path)),
                "type": "dir" if item.is_dir() else "file"
            })
        
        return {"entries": sorted(entries, key=lambda x: x["name"])}
```

**Config** (`config/tools/list_dir.json`):
```json
{
  "name": "list_dir",
  "implementation": "external.tools.parquet_agent.list_dir.ListDirTool",
  "description": "List files and directories in a domain folder",
  "version": "1.0.0",
  "enabled": true,
  "data_directory": "mock_datawarehouse"
}
```

---

### 3.2 `inspect_table` (Adapt from `duckdb_describe_table`)

**Purpose**: Get schema for a parquet/CSV file.

**Input Schema**:
```json
{
  "args": {
    "type": "object",
    "required": ["path"],
    "properties": {
      "path": { "type": "string", "description": "File path (e.g., 'ECommerce/sample_sales_data.csv')" }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["columns"],
    "properties": {
      "columns": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["name", "type"],
          "properties": {
            "name": { "type": "string" },
            "type": { "type": "string" }
          }
        }
      },
      "row_count_estimate": { "type": ["integer", "null"] },
      "primary_key_candidates": { "type": "array", "items": { "type": "string" } },
      "time_column_candidates": { "type": "array", "items": { "type": "string" } }
    }
  }
}
```

**Implementation**:
```python
# File: external/tools/parquet_agent/inspect_table.py

import duckdb
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool

class InspectTableTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "inspect_table",
            "description": "Get schema information for a data file",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.base_path = Path(self.config.get("data_directory", "mock_datawarehouse"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["columns"],
            "properties": {
                "columns": {"type": "array"},
                "row_count_estimate": {"type": ["integer", "null"]},
                "primary_key_candidates": {"type": "array"},
                "time_column_candidates": {"type": "array"}
            }
        }

    def execute(self, arguments):
        path = arguments["path"]
        full_path = self.base_path / path
        
        if not full_path.exists():
            raise ValueError(f"File not found: {path}")
        
        conn = duckdb.connect(":memory:")
        
        # Detect file type and read
        if full_path.suffix == ".parquet":
            query = f"DESCRIBE SELECT * FROM read_parquet('{full_path}')"
            count_query = f"SELECT COUNT(*) FROM read_parquet('{full_path}')"
        else:  # CSV
            query = f"DESCRIBE SELECT * FROM read_csv_auto('{full_path}')"
            count_query = f"SELECT COUNT(*) FROM read_csv_auto('{full_path}')"
        
        # Get schema
        schema_result = conn.execute(query).fetchall()
        columns = [{"name": row[0], "type": row[1]} for row in schema_result]
        
        # Get row count
        row_count = conn.execute(count_query).fetchone()[0]
        
        # Heuristics for key/time columns
        pk_candidates = [c["name"] for c in columns if "_id" in c["name"].lower() or c["name"].lower() == "id"]
        time_candidates = [c["name"] for c in columns if "date" in c["name"].lower() or "time" in c["name"].lower()]
        
        conn.close()
        
        return {
            "columns": columns,
            "row_count_estimate": row_count,
            "primary_key_candidates": pk_candidates,
            "time_column_candidates": time_candidates
        }
```

---

### 3.3 `preview_rows` (**NEW**)

**Purpose**: Sample rows for grain/filter intuition.

**Input Schema**:
```json
{
  "args": {
    "type": "object",
    "required": ["path", "limit"],
    "properties": {
      "path": { "type": "string" },
      "limit": { "type": "integer", "minimum": 1, "maximum": 100 }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["rows_preview"],
    "properties": {
      "rows_preview": { "type": "array", "items": { "type": "object" } }
    }
  }
}
```

**Implementation**:
```python
# File: external/tools/parquet_agent/preview_rows.py

import duckdb
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool

class PreviewRowsTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "preview_rows",
            "description": "Sample rows from a data file",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.base_path = Path(self.config.get("data_directory", "mock_datawarehouse"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["path", "limit"],
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["rows_preview"],
            "properties": {
                "rows_preview": {"type": "array", "items": {"type": "object"}}
            }
        }

    def execute(self, arguments):
        path = arguments["path"]
        limit = min(arguments.get("limit", 10), 100)
        full_path = self.base_path / path
        
        if not full_path.exists():
            raise ValueError(f"File not found: {path}")
        
        conn = duckdb.connect(":memory:")
        
        if full_path.suffix == ".parquet":
            query = f"SELECT * FROM read_parquet('{full_path}') LIMIT {limit}"
        else:
            query = f"SELECT * FROM read_csv_auto('{full_path}') LIMIT {limit}"
        
        df = conn.execute(query).fetchdf()
        conn.close()
        
        # Convert to list of dicts
        rows = df.to_dict(orient="records")
        
        return {"rows_preview": rows}
```

---

### 3.4 `search_glossary` (**NEW**)

**Purpose**: Search domain glossary for term definitions.

**Input Schema**:
```json
{
  "args": {
    "type": "object",
    "required": ["term", "domain"],
    "properties": {
      "term": { "type": "string" },
      "domain": { "type": "string" }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["hits"],
    "properties": {
      "hits": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["title", "snippet", "source_ref"],
          "properties": {
            "title": { "type": "string" },
            "snippet": { "type": "string" },
            "source_ref": { "type": "string" }
          }
        }
      }
    }
  }
}
```

**Implementation**:
```python
# File: external/tools/parquet_agent/search_glossary.py

import re
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool

class SearchGlossaryTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "search_glossary",
            "description": "Search domain glossary for term definitions",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.domain_dir = Path(self.config.get("domain_directory", "domain_instructions"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["term", "domain"],
            "properties": {
                "term": {"type": "string"},
                "domain": {"type": "string"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["hits"],
            "properties": {
                "hits": {"type": "array"}
            }
        }

    def execute(self, arguments):
        term = arguments["term"].lower()
        domain = arguments["domain"]
        
        # Find domain file
        domain_file = self.domain_dir / f"{domain}_domain.md"
        if not domain_file.exists():
            return {"hits": [], "error": f"Domain file not found: {domain}"}
        
        content = domain_file.read_text()
        hits = []
        
        # Parse metrics section
        metrics_match = re.search(r"## 5\) Metric dictionary.*?(?=## \d|\Z)", content, re.DOTALL)
        if metrics_match:
            metrics_text = metrics_match.group(0)
            # Find term in metrics
            for match in re.finditer(rf"- metric_name:\s*(\S+).*?(?=- metric_name:|\Z)", metrics_text, re.DOTALL):
                metric_block = match.group(0)
                metric_name = match.group(1)
                if term in metric_name.lower() or term in metric_block.lower():
                    definition_match = re.search(r"definition:\s*(.+?)(?=\n\s+-|\Z)", metric_block)
                    definition = definition_match.group(1) if definition_match else "No definition"
                    hits.append({
                        "title": metric_name,
                        "snippet": definition.strip(),
                        "source_ref": f"{domain}_domain.md#metrics"
                    })
        
        # Parse entities section
        entities_match = re.search(r"## 4\) Core entities.*?(?=## \d|\Z)", content, re.DOTALL)
        if entities_match:
            entities_text = entities_match.group(0)
            if term in entities_text.lower():
                for match in re.finditer(r"- name:\s*(\S+).*?typical_grain:\s*(.+?)(?=\n\s+-|\Z)", entities_text, re.DOTALL):
                    entity_name = match.group(1)
                    grain = match.group(2)
                    if term in entity_name.lower():
                        hits.append({
                            "title": entity_name,
                            "snippet": f"Grain: {grain.strip()}",
                            "source_ref": f"{domain}_domain.md#entities"
                        })
        
        return {"hits": hits}
```

---

### 3.5 `nl_to_sql_planner` (Rewrite)

**Purpose**: Convert completed QuerySpec to SQL.

> **CRITICAL**: The existing `nl_to_sql_planner.py` has wrong signature. Must rewrite to spec.

**Input Schema** (spec-compliant):
```json
{
  "args": {
    "type": "object",
    "required": ["query_spec", "query_spec_status"],
    "properties": {
      "query_spec": { "$ref": "query_spec.schema.json" },
      "query_spec_status": { "$ref": "query_spec_status.schema.json" }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["sql"],
    "properties": {
      "sql": { "type": "string" }
    }
  }
}
```

**Implementation**:
```python
# File: external/tools/parquet_agent/nl_to_sql_planner.py

from tools.base_mcp_tool import BaseMCPTool
from external.platform.llm import get_llm_client

NL_TO_SQL_PROMPT = """
ROLE
You are the SQL Generator.
Your job is to convert a completed Query Spec into SQL.

HARD CONSTRAINTS
- Output SQL ONLY. No prose, no markdown.
- You MUST NOT invent tables, columns, joins, metrics, or filters.
- You MUST NOT guess missing fields.
- You MUST NOT modify the Query Spec.
- If the Query Spec is incomplete for SQL generation, output exactly:
  ERROR: QUERY_SPEC_INCOMPLETE

INPUTS
Query Spec:
{query_spec}

Query Spec Status:
{query_spec_status}

SQL RULES
- Use query_spec.start_table.path as the base FROM source.
- Apply query_spec.filters exactly.
- Implement query_spec.aggregation_plan and preserve query_spec.grain.
- Apply query_spec.time.rule and query_spec.time.n_days when applicable.
- Respect query_spec.performance_guardrails.

OUTPUT
Return a single SQL string.
"""

class NLToSQLPlannerTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "nl_to_sql_planner",
            "description": "Convert QuerySpec to SQL",
            "version": "2.0.0",  # New version for spec compliance
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.llm_client = get_llm_client()

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["query_spec", "query_spec_status"],
            "properties": {
                "query_spec": {"type": "object"},
                "query_spec_status": {"type": "object"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["sql"],
            "properties": {
                "sql": {"type": "string"}
            }
        }

    def execute(self, arguments):
        import json
        query_spec = arguments["query_spec"]
        query_spec_status = arguments["query_spec_status"]
        
        # Check for blocking gaps
        for field, status in query_spec_status.items():
            if status.get("blocks_execution") and status.get("status") in ["missing", "conflict"]:
                return {"sql": "ERROR: QUERY_SPEC_INCOMPLETE"}
        
        prompt = NL_TO_SQL_PROMPT.format(
            query_spec=json.dumps(query_spec, indent=2),
            query_spec_status=json.dumps(query_spec_status, indent=2)
        )
        
        response = self.llm_client.complete(prompt)
        sql = response.strip()
        
        # Clean markdown if present
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        return {"sql": sql}
```

---

### 3.6 `sql_plan_updater` (**NEW**)

**Purpose**: Patch SQL based on error feedback.

**Input Schema**:
```json
{
  "args": {
    "type": "object",
    "required": ["sql", "patch_instructions"],
    "properties": {
      "sql": { "type": "string" },
      "patch_instructions": { "type": "string" }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["sql"],
    "properties": {
      "sql": { "type": "string" }
    }
  }
}
```

**Implementation**:
```python
# File: external/tools/parquet_agent/sql_plan_updater.py

from tools.base_mcp_tool import BaseMCPTool
from external.platform.llm import get_llm_client

SQL_PATCH_PROMPT = """
ROLE
You are the SQL Patcher.
Your job is to apply a minimal fix to the provided SQL.

HARD CONSTRAINTS
- Output SQL ONLY. No prose, no markdown.
- Apply ONLY the requested fix.
- Do NOT add features or change logic beyond the fix.
- Preserve the original intent and structure.

ORIGINAL SQL:
{sql}

PATCH INSTRUCTIONS:
{patch_instructions}

OUTPUT
Return the patched SQL string.
"""

class SQLPlanUpdaterTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "sql_plan_updater",
            "description": "Apply minimal patches to SQL",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.llm_client = get_llm_client()

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["sql", "patch_instructions"],
            "properties": {
                "sql": {"type": "string"},
                "patch_instructions": {"type": "string"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["sql"],
            "properties": {
                "sql": {"type": "string"}
            }
        }

    def execute(self, arguments):
        sql = arguments["sql"]
        patch_instructions = arguments["patch_instructions"]
        
        prompt = SQL_PATCH_PROMPT.format(
            sql=sql,
            patch_instructions=patch_instructions
        )
        
        response = self.llm_client.complete(prompt)
        patched_sql = response.strip()
        
        # Clean markdown if present
        if patched_sql.startswith("```"):
            patched_sql = patched_sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        return {"sql": patched_sql}
```

---

### 3.7 `query_safety_validator` (Adapt)

**Purpose**: Validate SQL for safety before execution.

> Existing tool at `external/tools/query_safety_validator.py` needs schema alignment.

**Changes Required**:
- Input: `query` → `sql`, add `policy_limits`
- Output: `is_safe` → `allowed`, `reason` → `flags`

**Implementation** (wrapper or edit existing):
```python
# Adapt existing or create wrapper
# Input: sql, policy_limits
# Output: allowed (bool), flags (array of strings)

def execute(self, arguments):
    sql = arguments["sql"]
    policy_limits = arguments.get("policy_limits", {})
    
    max_rows = policy_limits.get("max_rows", 1000)
    allow_cross_join = policy_limits.get("allow_cross_join", False)
    
    flags = []
    allowed = True
    
    # Check forbidden operations
    sql_upper = sql.upper()
    for op in ["DELETE", "UPDATE", "INSERT", "DROP", "TRUNCATE", "CREATE", "ALTER"]:
        if op in sql_upper:
            flags.append(f"FORBIDDEN_OP:{op}")
            allowed = False
    
    # Check cross join
    if "CROSS JOIN" in sql_upper and not allow_cross_join:
        flags.append("CROSS_JOIN_BLOCKED")
        allowed = False
    
    # Check for LIMIT
    if "LIMIT" not in sql_upper:
        flags.append(f"NO_LIMIT:will_enforce_{max_rows}")
    
    return {"allowed": allowed, "flags": flags}
```

---

### 3.8 `execute_sql` (Adapt from `duckdb_query`)

**Purpose**: Execute SQL and return results.

**Input Schema**:
```json
{
  "args": {
    "type": "object",
    "required": ["sql"],
    "properties": {
      "sql": { "type": "string" },
      "timeout_seconds": { "type": ["integer", "null"] },
      "max_rows": { "type": ["integer", "null"] }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["row_count", "columns", "rows_preview"],
    "properties": {
      "row_count": { "type": "integer" },
      "columns": { "type": "array", "items": { "type": "string" } },
      "rows_preview": { "type": "array", "items": { "type": "object" } }
    }
  }
}
```

**Implementation**:
```python
# File: external/tools/parquet_agent/execute_sql.py

import duckdb
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool

class ExecuteSQLTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "execute_sql",
            "description": "Execute SQL query on data warehouse",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.base_path = Path(self.config.get("data_directory", "mock_datawarehouse"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["sql"],
            "properties": {
                "sql": {"type": "string"},
                "timeout_seconds": {"type": ["integer", "null"]},
                "max_rows": {"type": ["integer", "null"]}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["row_count", "columns", "rows_preview"],
            "properties": {
                "row_count": {"type": "integer"},
                "columns": {"type": "array"},
                "rows_preview": {"type": "array"}
            }
        }

    def execute(self, arguments):
        sql = arguments["sql"]
        max_rows = arguments.get("max_rows", 100) or 100
        
        conn = duckdb.connect(":memory:")
        
        # Register data files as views
        for domain_dir in self.base_path.iterdir():
            if domain_dir.is_dir():
                for file in domain_dir.glob("*.csv"):
                    view_name = file.stem
                    conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{file}')")
                for file in domain_dir.glob("*.parquet"):
                    view_name = file.stem
                    conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{file}')")
        
        # Execute query
        result = conn.execute(sql)
        df = result.fetchdf()
        
        row_count = len(df)
        columns = list(df.columns)
        rows_preview = df.head(max_rows).to_dict(orient="records")
        
        conn.close()
        
        return {
            "row_count": row_count,
            "columns": columns,
            "rows_preview": rows_preview
        }
```

---

### 3.9 `query_result_evaluator` (Adapt)

**Purpose**: Evaluate if results satisfy QuerySpec.

**Input Schema**:
```json
{
  "args": {
    "type": "object",
    "required": ["query_spec", "results_summary", "validation_checks"],
    "properties": {
      "query_spec": { "$ref": "query_spec.schema.json" },
      "results_summary": { "type": "object" },
      "validation_checks": { "type": "array" }
    }
  }
}
```

**Output Schema**:
```json
{
  "returns": {
    "type": "object",
    "required": ["satisfied", "issues", "notes"],
    "properties": {
      "satisfied": { "type": "boolean" },
      "issues": { "type": "array", "items": { "type": "string" } },
      "notes": { "type": "string" }
    }
  }
}
```

**Implementation**: See existing `external/tools/query_result_evaluator.py`, align output schema.

---

## 4. Domain Configuration

### 4.1 ECommerce Domain (`domain_instructions/ecomm_domain.md`)

Create this file with the following content:

```markdown
# Domain: ecomm

## 1) Domain identity
- domain_key: ecomm
- description: ECommerce warehouse covering customers, inventory, and sales.

## 2) Time semantics (Decider reference)
- default_time_column: order_date
- default_time_rule: last_n_days
- default_time_n_days: 30
- supports_no_time_queries: true

## 3) Listing rules
- listing_allows_empty_metrics: true
- listing_default_limit: 50

## 4) Core entities (optional hints)
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

## 5) Metric dictionary (Decider reference; Executor may verify via glossary/schema)
- metrics:
  - metric_name: revenue
    definition: Sum of (quantity * price) from sales.
    default_filters: []
    required_tables: ["sample_sales_data"]
    disallowed_grains: []
  - metric_name: order_count
    definition: Count of distinct order_id.
    default_filters: []
    required_tables: ["sample_sales_data"]
    disallowed_grains: []
  - metric_name: total_purchases
    definition: Sum of total_purchases from customers table.
    default_filters: []
    required_tables: ["sample_customer_data"]
    disallowed_grains: []
  - metric_name: avg_order_value
    definition: revenue / order_count.
    default_filters: []
    required_tables: ["sample_sales_data"]
    disallowed_grains: []

## 6) Join conventions (optional hints)
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

## 7) Safety defaults (Executor reference)
- performance_guardrails:
  - default_limit: 50
  - avoid_select_star: true
  - allow_cross_join: false

## 8) Notes
- notes: Use order_date as the default time column for sales queries. Customer queries may use signup_date.
```

---

## 5. File Structure

After implementation, the tool files should be organized as:

```
external/tools/parquet_agent/
├── __init__.py
├── list_dir.py
├── inspect_table.py
├── preview_rows.py
├── search_glossary.py
├── nl_to_sql_planner.py      # Rewritten (v2)
├── sql_plan_updater.py       # NEW
├── query_safety_validator.py # Adapted
├── execute_sql.py
└── query_result_evaluator.py # Adapted

config/tools/
├── list_dir.json
├── inspect_table.json
├── preview_rows.json
├── search_glossary.json
├── nl_to_sql_planner.json    # Updated
├── sql_plan_updater.json     # NEW
├── query_safety_validator.json # Updated
├── execute_sql.json
└── query_result_evaluator.json # Updated
```

---

## 6. Testing Checklist

### Per-Tool Tests

| Tool | Test Case |
|------|-----------|
| `list_dir` | `{"path": "ECommerce"}` → returns 3 files |
| `inspect_table` | `{"path": "ECommerce/sample_sales_data.csv"}` → columns include order_id, customer_id |
| `preview_rows` | `{"path": "ECommerce/sample_sales_data.csv", "limit": 5}` → 5 rows |
| `search_glossary` | `{"term": "revenue", "domain": "ecomm"}` → hit with definition |
| `nl_to_sql_planner` | Valid QuerySpec → SQL string |
| `sql_plan_updater` | SQL + "add WHERE region='North'" → patched SQL |
| `query_safety_validator` | SELECT → allowed=true; DELETE → allowed=false |
| `execute_sql` | `{"sql": "SELECT * FROM sample_sales_data LIMIT 5"}` → 5 rows |
| `query_result_evaluator` | Results + QuerySpec → satisfied=true/false |

---

## 7. Registration

Register tools in `tools/tools_registry.py` or via config files:

```python
# In ToolsRegistry or agent initialization
from external.tools.parquet_agent import (
    ListDirTool,
    InspectTableTool,
    PreviewRowsTool,
    SearchGlossaryTool,
    NLToSQLPlannerTool,
    SQLPlanUpdaterTool,
    QuerySafetyValidatorTool,
    ExecuteSQLTool,
    QueryResultEvaluatorTool
)

PARQUET_AGENT_TOOLS = [
    ListDirTool,
    InspectTableTool,
    PreviewRowsTool,
    SearchGlossaryTool,
    NLToSQLPlannerTool,
    SQLPlanUpdaterTool,
    QuerySafetyValidatorTool,
    ExecuteSQLTool,
    QueryResultEvaluatorTool
]
```

---

## 8. Next Steps

1. Create `external/tools/parquet_agent/` directory
2. Implement each tool class
3. Create config JSON files
4. Populate `domain_instructions/ecomm_domain.md`
5. Run per-tool tests
6. Integrate with Executor in `parquet_agent.py`

