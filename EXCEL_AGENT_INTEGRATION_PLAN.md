# Excel Agent Integration Plan

## Overview

Integrate the Excel Agent as a configurable agent type that works with both BMO Financials and ECommerce use cases through UI configuration.

---

## 1. Tools Required

### Core Excel Tools (Already Created)

**Location:** `external/tools/excel_agent/`

1. **`list_excel_files.py`** - `ListExcelFilesTool`
   - Lists all Excel files (.xlsx, .xls) in data folder
   - Input: `agent_data_folder` (optional), `path` (optional)
   - Output: Array of files with name, path, size_bytes

2. **`read_excel_sheet.py`** - `ReadExcelSheetTool`
   - Reads Excel file and lists sheets OR reads specific sheet data
   - Input: `file_path`, `sheet_name` (optional), `max_rows` (optional), `agent_data_folder`
   - Output: Sheets list OR data rows with columns

3. **`query_excel_data.py`** - `QueryExcelDataTool`
   - Executes data operations (filter, group, aggregate, sort, etc.)
   - Input: `file_path`, `sheet_name`, `operation`, filters/aggregations/etc.
   - Output: Query results as data rows

### Tool Configuration Files Needed

**Location:** `external/config/tools/`

Create JSON configs for each tool:

```json
// list_excel_files.json
{
  "name": "list_excel_files",
  "description": "List Excel files in data folder",
  "version": "1.0.0",
  "enabled": true,
  "class_path": "external.tools.excel_agent.list_excel_files.ListExcelFilesTool",
  "config": {
    "data_directory": "external/datawarehouse"
  }
}
```

```json
// read_excel_sheet.json
{
  "name": "read_excel_sheet",
  "description": "Read Excel sheet data",
  "version": "1.0.0",
  "enabled": true,
  "class_path": "external.tools.excel_agent.read_excel_sheet.ReadExcelSheetTool",
  "config": {
    "data_directory": "external/datawarehouse"
  }
}
```

```json
// query_excel_data.json
{
  "name": "query_excel_data",
  "description": "Query Excel data with operations",
  "version": "1.0.0",
  "enabled": true,
  "class_path": "external.tools.excel_agent.query_excel_data.QueryExcelDataTool",
  "config": {
    "data_directory": "external/datawarehouse"
  }
}
```

---

## 2. Prompts/Domain Files

### Domain Markdown Files (Configuration)

**Location:** `external/config/domains/`

#### A) BMO Financials Domain
**File:** `bmo_financials_domain.md` ✅ (Already Created)

**Key Sections:**
- File naming conventions (Suppq323 = Q3 2023)
- Sheet structure (39+ sheets, Index navigation)
- Content types by page number
- Query strategies
- Key metrics locations

#### B) ECommerce Domain  
**File:** `ecommerce_excel_domain.md` (Need to Create)

**Should Include:**
- File structure (CSV converted to Excel)
- Entity relationships (customers, products, sales)
- Column mappings
- Query patterns for sales analysis

### Domain File Template Structure

```markdown
# Domain: {domain_name}

## 1) Domain Identity
- domain_key: {key}
- description: {description}

## 2) File Structure
- File naming patterns
- Sheet organization
- Data layout

## 3) Data Characteristics
- Currency/units
- Format challenges
- Data quality notes

## 4) Content Types & Locations
- What data is where
- Key pages/sheets
- Metric locations

## 5) Query Strategy
- How to navigate files
- Common query patterns
- Best practices

## 6) Key Metrics
- Important metrics to track
- Where to find them
- Calculation methods
```

---

## 3. Agent Configuration Structure

### Agent Registry Entry

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
    "max_rows_per_sheet": 100
  }
}
```

### Excel Agent-Specific Config

```json
{
  "id": "ecommerce_excel_agent",
  "name": "ECommerce Excel Agent",
  "description": "Analyze ecommerce sales data from Excel files",
  "agent_type": "excel",
  "domain_file": "ecommerce_excel_domain.md",
  "data_folder": "ECommerce",
  "config": {
    "read_all_sheets": false,  // For simple Excel files
    "max_iterations": 10,
    "max_rows_per_sheet": 50
  }
}
```

---

## 4. Integration Points

### A) Base Agent Wrapper

**File:** `external/agent/excel_base_agent.py` (New)

Create a unified Excel agent that:
- Loads domain file from agent config
- Uses domain knowledge to guide queries
- Supports both simple (ECommerce) and complex (BMO) Excel structures
- Reads all sheets when configured

```python
class ExcelBaseAgent:
    def __init__(self, agent_config: Dict):
        self.agent_config = agent_config
        self.domain_file = agent_config.get("domain_file")
        self.data_folder = agent_config.get("data_folder")
        self.load_domain_knowledge()
        
    def handle_query(self, user_query: str, ...) -> Dict:
        # Unified ReAct loop
        # Uses domain knowledge for planning
        # Reads all sheets if configured
```

### B) API Route Integration

**File:** `external/routes/agent_run_routes.py` (Update)

Add Excel agent handling:

```python
@app.route('/api/agent/run', methods=['POST'])
def run_agent():
    agent_id = request.json.get('agent_id')
    query = request.json.get('query')
    
    agent = get_agent(agent_id)
    agent_type = agent.get('agent_type')
    
    if agent_type == 'excel':
        from external.agent.excel_base_agent import ExcelBaseAgent
        excel_agent = ExcelBaseAgent(agent)
        result = excel_agent.handle_query(query, ...)
        return jsonify(result)
```

### C) MCP Tool Integration

**File:** `external/agent/excel_mcp_tool.py` (New)

Create MCP-facing wrapper similar to `LangGraphAgentTool`:

```python
class ExcelAgentTool(BaseMCPTool):
    def __init__(self, config: Dict = None):
        # MCP tool wrapper for Excel agent
        # Handles sessions, conversation history
        # Returns standardized response format
```

---

## 5. UI Configuration Requirements

### A) Agent Creation Form

**Location:** `external/web/templates/admin.html` or new template

**Fields Needed:**
1. **Agent Type Selection**
   - Dropdown: "Excel Agent"
   
2. **Basic Info**
   - Name
   - Description
   
3. **Data Source Configuration**
   - Data Folder (dropdown or text input)
   - Domain File (dropdown of available domain files OR text editor)
   
4. **Excel-Specific Settings**
   - ☑ Read All Sheets (checkbox)
   - Max Iterations (number input, default: 10)
   - Max Rows Per Sheet (number input, default: 100)

5. **Domain Knowledge Editor**
   - Text area for domain markdown
   - OR: Select existing domain file
   - Preview domain file

### B) Agent Selection Page

**Location:** `external/web/templates/agents.html`

**Display:**
- Group Excel agents separately
- Show data folder and domain file
- Indicate "Read All Sheets" capability

### C) Agent Chat UI

**Location:** `external/web/templates/agent_chat.html`

**Excel-Specific Features:**
- Show which sheets were read
- Display file structure
- Show micro-summaries
- Export results to Excel

---

## 6. Database Schema (If Using DB)

**Table:** `agents` (if exists)

**Excel-Specific Columns:**
- `agent_type`: "excel"
- `domain_file`: Path to domain markdown
- `data_folder`: Data warehouse folder
- `config`: JSON with Excel-specific settings

---

## 7. Implementation Steps

### Phase 1: Core Integration
1. ✅ Create Excel tools (DONE)
2. ✅ Create domain files (BMO done, ECommerce needed)
3. Create unified `ExcelBaseAgent` class
4. Create MCP tool wrapper
5. Register tools in ToolsRegistry

### Phase 2: API Integration
1. Update `agent_run_routes.py` to handle Excel agent type
2. Add Excel agent to agent selection
3. Test query execution through API

### Phase 3: UI Integration
1. Add Excel agent type to agent creation form
2. Add Excel-specific configuration fields
3. Update agent selection page
4. Test end-to-end flow

### Phase 4: Domain Configuration
1. Create ECommerce Excel domain file
2. Test both BMO and ECommerce agents
3. Document domain file creation process

---

## 8. Configuration Examples

### Example 1: BMO Financials Agent

```json
{
  "id": "bmo_financials_agent",
  "name": "BMO Financials Agent",
  "description": "Analyze quarterly supplement reports",
  "agent_type": "excel",
  "domain_file": "bmo_financials_domain.md",
  "data_folder": "bmo_financials",
  "config": {
    "read_all_sheets": true,
    "max_iterations": 15,
    "max_rows_per_sheet": 100,
    "always_read_index": true
  }
}
```

### Example 2: ECommerce Excel Agent

```json
{
  "id": "ecommerce_excel_agent",
  "name": "ECommerce Excel Agent",
  "description": "Analyze sales and customer data",
  "agent_type": "excel",
  "domain_file": "ecommerce_excel_domain.md",
  "data_folder": "ECommerce",
  "config": {
    "read_all_sheets": false,
    "max_iterations": 10,
    "max_rows_per_sheet": 50
  }
}
```

---

## 9. Key Design Decisions

### A) Unified vs Separate Agents

**Decision:** Create unified `ExcelBaseAgent` that:
- Loads domain file dynamically
- Adapts behavior based on domain knowledge
- Supports both simple and complex Excel structures

**Rationale:** 
- Single codebase to maintain
- Domain files provide flexibility
- Easier to add new Excel use cases

### B) Read All Sheets Strategy

**Decision:** Make it configurable per agent
- BMO: `read_all_sheets: true` (complex, many sheets)
- ECommerce: `read_all_sheets: false` (simple, few sheets)

**Rationale:**
- Performance optimization
- Different use cases have different needs

### C) Domain File Location

**Decision:** Store in `external/config/domains/`
- Same location as other domain files
- Referenced by `domain_file` in agent config

---

## 10. Testing Checklist

- [ ] Create BMO agent through UI
- [ ] Create ECommerce agent through UI
- [ ] Execute query on BMO agent
- [ ] Execute query on ECommerce agent
- [ ] Verify all sheets read for BMO
- [ ] Verify correct domain knowledge loaded
- [ ] Test session continuity
- [ ] Test error handling
- [ ] Verify data folder scoping works
- [ ] Test with multiple Excel files

---

## 11. Files to Create/Modify

### New Files
1. `external/agent/excel_base_agent.py` - Unified Excel agent
2. `external/agent/excel_mcp_tool.py` - MCP wrapper
3. `external/config/domains/ecommerce_excel_domain.md` - ECommerce domain
4. `external/config/tools/list_excel_files.json` - Tool config
5. `external/config/tools/read_excel_sheet.json` - Tool config
6. `external/config/tools/query_excel_data.json` - Tool config

### Modified Files
1. `external/agent/agent_registry.py` - Already supports "excel" type ✅
2. `external/routes/agent_run_routes.py` - Add Excel agent handling
3. `external/web/templates/admin.html` - Add Excel agent creation form
4. `external/web/templates/agents.html` - Display Excel agents
5. `tools/tools_registry.py` - Register Excel tools

---

## 12. Next Steps

1. **Create unified ExcelBaseAgent** - Merge BMO and generic Excel agents
2. **Create ECommerce domain file** - Based on existing ECommerce structure
3. **Create tool configs** - JSON configs for ToolsRegistry
4. **Update API routes** - Add Excel agent execution path
5. **Update UI** - Add Excel agent configuration form
6. **Test end-to-end** - Both BMO and ECommerce use cases

---

**Status:** Ready for implementation
**Priority:** High
**Estimated Effort:** 2-3 days

