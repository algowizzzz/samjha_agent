# Excel Agent Integration: Tools & Prompts

## Summary

This document outlines the **tools** and **prompts/domain files** needed to integrate the Excel Agent into the application, making it configurable for both BMO Financials and ECommerce use cases.

---

## 1. Tools Required

### A) Excel Tools (Already Created ✅)

**Location:** `external/tools/excel_agent/`

| Tool | File | Purpose |
|------|------|---------|
| `ListExcelFilesTool` | `list_excel_files.py` | List all Excel files in data folder |
| `ReadExcelSheetTool` | `read_excel_sheet.py` | Read Excel sheets and data |
| `QueryExcelDataTool` | `query_excel_data.py` | Execute queries (filter, group, aggregate) |

### B) Tool Configuration Files (Need to Create)

**Location:** `external/config/tools/`

#### 1. `list_excel_files.json`
```json
{
  "name": "list_excel_files",
  "description": "List Excel files (.xlsx, .xls) in a data folder",
  "version": "1.0.0",
  "enabled": true,
  "class_path": "external.tools.excel_agent.list_excel_files.ListExcelFilesTool",
  "config": {
    "data_directory": "external/datawarehouse"
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "agent_data_folder": {
        "type": "string",
        "description": "Optional agent-specific data folder"
      },
      "path": {
        "type": "string",
        "description": "Optional subdirectory path"
      }
    }
  }
}
```

#### 2. `read_excel_sheet.json`
```json
{
  "name": "read_excel_sheet",
  "description": "Read Excel file and list sheets or read specific sheet data",
  "version": "1.0.0",
  "enabled": true,
  "class_path": "external.tools.excel_agent.read_excel_sheet.ReadExcelSheetTool",
  "config": {
    "data_directory": "external/datawarehouse"
  },
  "inputSchema": {
    "type": "object",
    "required": ["file_path"],
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to Excel file relative to datawarehouse"
      },
      "sheet_name": {
        "type": "string",
        "description": "Optional sheet name. If not provided, returns list of sheets"
      },
      "max_rows": {
        "type": "integer",
        "description": "Maximum rows to read (default: 100)",
        "default": 100
      },
      "agent_data_folder": {
        "type": "string",
        "description": "Optional agent-specific data folder"
      }
    }
  }
}
```

#### 3. `query_excel_data.json`
```json
{
  "name": "query_excel_data",
  "description": "Execute data operations (filter, group, aggregate, sort) on Excel data",
  "version": "1.0.0",
  "enabled": true,
  "class_path": "external.tools.excel_agent.query_excel_data.QueryExcelDataTool",
  "config": {
    "data_directory": "external/datawarehouse"
  },
  "inputSchema": {
    "type": "object",
    "required": ["file_path", "sheet_name", "operation"],
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to Excel file"
      },
      "sheet_name": {
        "type": "string",
        "description": "Sheet name to query"
      },
      "operation": {
        "type": "string",
        "enum": ["filter", "group_by", "aggregate", "sort", "select_columns", "head", "tail", "describe"],
        "description": "Operation to perform"
      },
      "filters": {
        "type": "object",
        "description": "Filter conditions"
      },
      "group_by_columns": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Columns to group by"
      },
      "aggregations": {
        "type": "object",
        "description": "Aggregation functions"
      },
      "max_rows": {
        "type": "integer",
        "description": "Maximum rows to return",
        "default": 1000
      },
      "agent_data_folder": {
        "type": "string",
        "description": "Optional agent-specific data folder"
      }
    }
  }
}
```

---

## 2. Prompts/Domain Files

### A) BMO Financials Domain ✅ (Already Created)

**File:** `external/config/domains/bmo_financials_domain.md`

**Key Sections:**
1. **Domain Identity** - domain_key, description
2. **File Structure** - Naming conventions (Suppq323 = Q3 2023), sheet organization
3. **Data Characteristics** - Currency (CAD millions), format challenges
4. **Content Types & Locations** - What's on each page (Page 5 = Financial Highlights, Page 24 = Credit Risk)
5. **Query Strategy** - How to navigate, always read Index first
6. **Key Metrics** - ROE, ROA, PCL, Capital ratios, where to find them
7. **File Identification** - Period mapping (q323 = Q3 2023)

**Usage:** Guides agent to:
- Read Index sheet first
- Navigate to correct pages based on query intent
- Understand BMO-specific structure (39+ sheets per file)

### B) ECommerce Excel Domain (Need to Create)

**File:** `external/config/domains/ecommerce_excel_domain.md`

**Should Include:**

```markdown
# Domain: ECommerce Excel

## 1) Domain Identity
- domain_key: ecommerce_excel
- description: ECommerce sales, customer, and inventory data in Excel format

## 2) File Structure
- Files: sample_sales_data.xlsx, sample_customer_data.xlsx, sample_inventory_data.xlsx
- Simple structure: One sheet per file (Sheet1)
- Data is tabular (not report-style like BMO)

## 3) Data Characteristics
- Currency: Dollars (USD)
- Format: Clean tabular data with named columns
- No Index sheet needed

## 4) Entity Relationships
- **Customers** (customer_id) ↔ **Sales** (customer_id)
- **Products** (product names) ↔ **Sales** (product)
- **Sales** contains: order_id, customer_id, product, category, quantity, price, order_date, region

## 5) Query Strategy
- Direct file reading (no Index navigation)
- Simple sheet access (usually Sheet1)
- Can join across files using customer_id or product names

## 6) Key Metrics
- Revenue (price × quantity)
- Customer lifetime value
- Product performance
- Regional sales
- Category distribution

## 7) Common Query Patterns
- "Revenue by customer" → Read sales, group by customer_id, sum(price × quantity)
- "Top products" → Read sales, group by product, sum quantity
- "Customer analysis" → Join sales + customer files on customer_id
```

---

## 3. Agent Configuration Structure

### Agent JSON Config Format

**Location:** `external/config/agents/{agent_id}.json`

```json
{
  "id": "bmo_financials_agent",
  "name": "BMO Financials Agent",
  "description": "Analyze BMO quarterly supplement reports",
  "agent_type": "excel",
  "domain_file": "bmo_financials_domain.md",
  "data_folder": "bmo_financials",
  "created_at": "2026-01-07T00:00:00Z",
  "updated_at": "2026-01-07T00:00:00Z",
  "config": {
    "read_all_sheets": true,
    "max_iterations": 15,
    "max_rows_per_sheet": 100,
    "always_read_index": true
  }
}
```

```json
{
  "id": "ecommerce_excel_agent",
  "name": "ECommerce Excel Agent",
  "description": "Analyze ecommerce sales data from Excel files",
  "agent_type": "excel",
  "domain_file": "ecommerce_excel_domain.md",
  "data_folder": "ECommerce",
  "created_at": "2026-01-07T00:00:00Z",
  "updated_at": "2026-01-07T00:00:00Z",
  "config": {
    "read_all_sheets": false,
    "max_iterations": 10,
    "max_rows_per_sheet": 50
  }
}
```

---

## 4. Integration Points

### A) Unified Excel Agent

**File:** `external/agent/excel_base_agent.py` (New)

**Key Features:**
- Loads domain file from agent config
- Adapts behavior based on domain knowledge
- Supports both simple (ECommerce) and complex (BMO) structures
- Configurable: read_all_sheets, max_iterations, etc.

**Interface:**
```python
def handle_excel_query(
    user_query: str,
    agent_id: Optional[str] = None,
    conversation_history: List[Dict] = None,
    prior_state: Optional[Dict] = None,
    show_thinking: bool = False,
) -> Dict[str, Any]:
    # Load agent config
    # Load domain file
    # Execute ReAct loop
    # Return standardized response
```

### B) Route Integration

**File:** `external/routes/agent_run_routes.py` (Modify)

**Add Excel case around line 633:**

```python
elif agent_type == "excel":
    # Excel agent
    from external.agent.excel_base_agent import handle_excel_query
    result = handle_excel_query(
        user_query=user_query,
        agent_id=agent_id,
        conversation_history=conversation_history,
        prior_state=None,  # Excel agent manages its own state
        show_thinking=show_thinking,
    )
```

---

## 5. Prompt/System Instructions

### A) Domain File as Prompt Context

The domain markdown file serves as the **system prompt** for the Excel agent:

- **BMO Domain** → Guides agent to read Index, navigate pages, understand financial structure
- **ECommerce Domain** → Guides agent to read files directly, understand relationships

### B) Agent-Specific Instructions

**Embedded in Excel Agent Logic:**

```python
# From domain file, extract:
- File naming patterns
- Sheet organization rules
- Query strategies
- Key metric locations
- Data characteristics

# Use in planning:
- "If BMO domain → always read Index first"
- "If ECommerce domain → read files directly"
- "If query mentions 'segment' → check Page 5-13 (BMO)"
- "If query mentions 'customer' → join customer + sales files (ECommerce)"
```

---

## 6. Current State vs Required State

### ✅ Already Done
- [x] Excel tools created (list, read, query)
- [x] BMO domain file created
- [x] BMO Excel agent created (specialized)
- [x] Generic Excel agent created
- [x] Agent registry supports "excel" type

### ❌ Need to Create
- [ ] Tool JSON configs (3 files)
- [ ] ECommerce Excel domain file
- [ ] Unified ExcelBaseAgent (merge BMO + generic)
- [ ] Route integration in agent_run_routes.py
- [ ] UI form for Excel agent creation
- [ ] MCP tool wrapper (optional)

---

## 7. Quick Start Checklist

### Step 1: Create Tool Configs
```bash
# Create 3 JSON files in external/config/tools/
- list_excel_files.json
- read_excel_sheet.json
- query_excel_data.json
```

### Step 2: Create ECommerce Domain
```bash
# Create ecommerce_excel_domain.md in external/config/domains/
# Based on existing ECommerce structure
```

### Step 3: Create Unified Agent
```bash
# Create excel_base_agent.py
# Merge BMO and generic Excel agents
# Load domain file dynamically
```

### Step 4: Integrate Routes
```bash
# Modify agent_run_routes.py
# Add "excel" case in routing logic
```

### Step 5: Test
```bash
# Create BMO agent through UI
# Create ECommerce agent through UI
# Test queries on both
```

---

## 8. Configuration Flow

```
User Creates Agent (UI)
    ↓
Agent Config Saved (JSON)
    ↓
Domain File Referenced (domain_file field)
    ↓
Excel Agent Loads Domain File
    ↓
Domain File Guides Query Planning
    ↓
Tools Execute (list, read, query)
    ↓
Results Synthesized with Domain Context
    ↓
Response Returned to UI
```

---

## 9. Key Design Decisions

### Decision 1: Unified vs Separate Agents
**Choice:** Unified `ExcelBaseAgent` with domain-driven behavior
**Rationale:** Single codebase, domain files provide flexibility

### Decision 2: Domain File Location
**Choice:** `external/config/domains/` (same as other domains)
**Rationale:** Consistent with existing structure

### Decision 3: Read All Sheets
**Choice:** Configurable per agent (`read_all_sheets: true/false`)
**Rationale:** Performance optimization, different use cases

### Decision 4: Tool Registration
**Choice:** JSON configs in `external/config/tools/`
**Rationale:** Consistent with existing tool registration pattern

---

## 10. Files Summary

### Tools (3 files) ✅
- `external/tools/excel_agent/list_excel_files.py`
- `external/tools/excel_agent/read_excel_sheet.py`
- `external/tools/excel_agent/query_excel_data.py`

### Tool Configs (3 files) ❌
- `external/config/tools/list_excel_files.json`
- `external/config/tools/read_excel_sheet.json`
- `external/config/tools/query_excel_data.json`

### Domain Files (2 files)
- `external/config/domains/bmo_financials_domain.md` ✅
- `external/config/domains/ecommerce_excel_domain.md` ❌

### Agent Files (1 file) ❌
- `external/agent/excel_base_agent.py` (unified)

### Route Integration (1 file) ❌
- `external/routes/agent_run_routes.py` (modify)

---

**Next Step:** Start with creating the tool configs and unified agent, then integrate into routes.

