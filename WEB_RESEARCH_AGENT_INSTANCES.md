# Web Research Agent Instances - Current Status

## Summary

**Current Web Research Agent Instances**: **0 confirmed in file registry** | **1 potential in database** (if setup script was run)

**Agent Type**: `"external"` (used for web research agents)

---

## Agent Storage Locations

The system uses **two storage mechanisms**:

1. **File-Based Registry** (`external/config/agents/*.json`)
   - Legacy/fallback storage
   - Currently contains **0 web research agents** (all 3 agents are "structured" type)

2. **Database Storage** (Primary)
   - Agents stored in `Agent` table
   - Supports web research agent fields: `tavily_api_key`, `search_scope_allowed_domains`, `search_scope_blocked_domains`, `default_research_depth`

---

## File-Based Registry Status

### Current Agents in File Registry

| Agent ID | Name | Type | Status |
|----------|------|------|--------|
| `ecommerce_agent` | Ecommerce Agent | `structured` | ✅ Active |
| `sales` | sales | `structured` | ✅ Active |
| `widget_sales_agent` | Widget Sales Agent | `structured` | ✅ Active |

**Total**: 3 agents, **0 web research agents**

---

## Database-Based Agents

### Setup Script Available

The system includes `setup_web_search_agent.py` which creates:

**Financial News Research Agent**
- **Agent ID**: `financial-news-research-agent` (slugified from name)
- **Agent Type**: `external` (web research)
- **Name**: "Financial News Research Agent"
- **Description**: "Web research agent focused on financial news, market trends, and economic indicators"
- **Domain File**: `financial_news_domain.md`
- **Model**: `claude-3-sonnet-20240229` (default)
- **Research Depth**: `standard`
- **Allowed Domains**: 
  - `sec.gov` (SEC)
  - `federalreserve.gov` (Federal Reserve)
  - `treasury.gov` (U.S. Treasury)
  - `bloomberg.com` (Bloomberg)
  - `reuters.com` (Reuters)
  - `wsj.com` (Wall Street Journal)
  - `ft.com` (Financial Times)
- **Blocked Domains**: 
  - `reddit.com`
  - `twitter.com`
  - `facebook.com`

### Domain Configuration

The Financial News Research Agent includes domain configuration for:
- Authority domains (government sites, major financial news)
- Research depth settings (2-3 iterations, 6-20 sources)
- Source quality requirements
- Search scope (allowed/blocked domains)
- Time range defaults (last 12 months)
- Research focus areas (market analysis, company performance, economic indicators, etc.)

---

## How to Check Existing Instances

### Via API

```bash
# List all agents
GET /api/admin/agents

# Filter for web research agents (agent_type="external")
# Response will include:
{
  "id": "financial-news-research-agent",
  "name": "Financial News Research Agent",
  "agent_type": "external",
  "description": "...",
  "tavily_api_key": "...",
  "search_scope_allowed_domains": [...],
  "search_scope_blocked_domains": [...],
  "default_research_depth": "standard"
}
```

### Via Setup Script

```bash
# Run setup script to create Financial News Research Agent
python setup_web_search_agent.py
```

This script:
1. Imports web search prompts to database
2. Creates Financial News Research Agent
3. Tests agent retrieval

---

## Creating New Web Research Agents

### Via Admin UI

1. Navigate to `/admin` → "Manage Agents" → "External Data Agents"
2. Click "+ Create New Agent"
3. Fill in:
   - **Name**: Agent display name
   - **Description**: Agent purpose
   - **Agent Type**: `external` (automatically set)
   - **Domain Content**: Markdown domain configuration
   - **Model**: LLM model to use
   - **Tavily API Key**: (optional, can be set later)
   - **Allowed Domains**: List of allowed domains
   - **Blocked Domains**: List of blocked domains
   - **Research Depth**: `standard` | `deep` | `quick`

### Via API

```bash
POST /api/admin/agents
Content-Type: multipart/form-data

{
  "name": "My Web Research Agent",
  "description": "Agent description",
  "agent_type": "external",
  "domain_file": "my_domain.md",
  "domain_content": "# Domain config...",
  "model": "claude-3-sonnet-20240229",
  "tavily_api_key": "optional",
  "search_scope_allowed_domains": ["example.com"],
  "search_scope_blocked_domains": ["spam.com"],
  "default_research_depth": "standard"
}
```

### Via Setup Script (Custom)

Modify `setup_web_search_agent.py` to create custom agents:

```python
def create_custom_agent():
    agent_name = "My Custom Research Agent"
    agent_id = slugify_name(agent_name)
    
    domain_content = """# Custom Domain Configuration
## Authority Domains
- example.com
- trusted-source.com
...
"""
    
    create_agent_db(
        db,
        agent_id=agent_id,
        name=agent_name,
        agent_type="external",
        description="Custom web research agent",
        domain_file="custom_domain.md",
        domain_content=domain_content,
        ...
    )
```

---

## Agent Type Detection

The system automatically detects agent type:

```python
# In agent_run_routes.py
agent_type = agent.get("agent_type", "structured")

if agent_type == "external":
    # Use web research agent handler
    from external.agent.web_research_agent import handle_web_research_query
    result = handle_web_research_query(...)
else:
    # Use structured agent handler
    from external.agent.parquet_agent import handle_query
    result = handle_query(...)
```

The decider output also indicates type:
- **Structured**: Has `query_spec` in decider output
- **Web Research**: Has `research_spec` in decider output

---

## Expected Web Research Agent Fields

When creating/querying web research agents, expect these fields:

| Field | Type | Description |
|-------|------|-------------|
| `agent_type` | `str` | Must be `"external"` |
| `tavily_api_key` | `str` | Tavily API key (encrypted in DB) |
| `search_scope_allowed_domains` | `List[str]` | Allowed domains for search |
| `search_scope_blocked_domains` | `List[str]` | Blocked domains |
| `default_research_depth` | `str` | `"standard"` \| `"deep"` \| `"quick"` |
| `domain_content` | `str` | Markdown domain configuration |

---

## Migration Status

The database schema includes web research agent fields via migration:
- `alembic/versions/0002_add_web_search_agent_fields.py`

This migration adds:
- `tavily_api_key` (encrypted)
- `search_scope_allowed_domains` (JSON)
- `search_scope_blocked_domains` (JSON)
- `default_research_depth` (string)

---

## Conclusion

**Current State**:
- ✅ Web research agent infrastructure is complete
- ✅ Setup script available (`setup_web_search_agent.py`)
- ✅ API endpoints support web research agents
- ⚠️ **No confirmed instances** in file registry
- ❓ **1 potential instance** in database (Financial News Research Agent, if setup script was run)

**To Verify**:
1. Run `python setup_web_search_agent.py` to create Financial News Research Agent
2. Query `/api/admin/agents` to list all agents
3. Filter for `agent_type="external"` to see web research agents

**To Create New Instances**:
- Use Admin UI (`/admin` → External Data Agents)
- Use API (`POST /api/admin/agents`)
- Modify setup script for custom agents

