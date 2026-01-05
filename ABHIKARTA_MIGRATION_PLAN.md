# Abhikarta Migration Plan

> **Target:** Migrate samjha_agent-1 → [abhikarta-llm](https://github.com/ajsinha/abhikarta-llm)
> **Status:** Planning
> **Last Updated:** January 4, 2026

---

## TLDR Checklist

### Phase 1: Pre-Migration (Week 1)
- [ ] Clone and run Abhikarta locally
- [ ] Review Abhikarta database schema (45 tables)
- [ ] Map existing samjha tables → Abhikarta tables
- [ ] Identify features to preserve vs. deprecate
- [ ] Get license/usage agreement from Abhikarta owner

### Phase 2: Database Migration (Week 2-3)
- [ ] Create schema mapping document
- [ ] Write migration scripts for core tables:
  - [ ] `agents` → Abhikarta `agents`
  - [ ] `prompts` / `agent_prompts` → Abhikarta templates
  - [ ] `conversations` / `messages` → Abhikarta equivalents
  - [ ] `runs` / `run_events` → Abhikarta `executions` / `execution_steps`
- [ ] Write migration scripts for bulk doc tables:
  - [ ] `sessions` / `documents` → Abhikarta document system
  - [ ] `chains` / `chain_steps` → Abhikarta workflows
  - [ ] `workflows` / `workflow_versions` → merge with Abhikarta workflows
  - [ ] `step_results` → Abhikarta execution results
- [ ] Create rollback scripts
- [ ] Test migrations on copy of prod data

### Phase 3: Tool Framework Migration (Week 3-4)
- [ ] Audit existing tools in `tools/` and `external/tools/`
- [ ] Create Abhikarta tool wrappers:
  - [ ] DuckDB/Parquet tools → Abhikarta `BaseTool` subclass
  - [ ] MCP tools → Abhikarta `MCPTool` integration
  - [ ] Web research tools → Abhikarta `HTTPTool`
- [ ] Register tools in Abhikarta `ToolsRegistry`
- [ ] Test tool execution via Abhikarta API

### Phase 4: Agent Migration (Week 4-5)
- [ ] Convert agent definitions:
  - [ ] `parquet_agent.py` → Abhikarta agent template
  - [ ] `web_research_agent.py` → Abhikarta agent template
  - [ ] `base_agent.py` logic → Abhikarta agent framework
- [ ] Migrate agent configs from `external/config/agents/`
- [ ] Migrate domain files from `external/config/domains/`
- [ ] Migrate prompts from `external/config/prompts/`
- [ ] Test agent execution in Abhikarta

### Phase 5: API & Routes Migration (Week 5-6)
- [ ] Map existing routes → Abhikarta routes:
  - [ ] Auth routes (`auth_routes.py`)
  - [ ] Agent routes (`agent_routes.py`)
  - [ ] Admin routes (`admin_routes.py`)
  - [ ] API routes (`api_routes.py`)
  - [ ] Tools routes (`tools_routes.py`)
- [ ] Create API compatibility shim (if needed for existing clients)
- [ ] Update frontend API calls
- [ ] Test all endpoints

### Phase 6: UI Migration (Week 6-8)
- [ ] Migrate templates:
  - [ ] `home.html` → Abhikarta dashboard
  - [ ] `agent_chat.html` → Abhikarta agent UI
  - [ ] `admin.html` → Abhikarta admin panel
  - [ ] `bulk_doc_analysis.html` → adapt or rebuild
- [ ] Migrate static assets (CSS/JS)
- [ ] Integrate with Abhikarta visual designer
- [ ] Test all UI flows

### Phase 7: New Features Setup (Week 8-9)
- [ ] Configure HITL (Human-in-the-loop) if needed
- [ ] Configure LLM providers in Abhikarta
- [ ] Set up notification system (if needed)
- [ ] Configure audit logging
- [ ] Set up API key management

### Phase 8: Testing & Validation (Week 9-10)
- [ ] Run existing test suites against Abhikarta
- [ ] Create integration tests for migrated components
- [ ] Performance testing
- [ ] Security review
- [ ] User acceptance testing

### Phase 9: Deployment (Week 10-11)
- [ ] Create deployment runbook
- [ ] Set up staging environment
- [ ] Deploy to staging
- [ ] Run smoke tests
- [ ] Deploy to production
- [ ] Monitor for issues

### Phase 10: Cleanup (Week 11-12)
- [ ] Archive old samjha codebase
- [ ] Update documentation
- [ ] Train team on Abhikarta
- [ ] Remove deprecated code
- [ ] Close migration project

---

## Detailed Migration Plan

### 1. Database Schema Mapping

#### 1.1 Core Agent Tables

| samjha Table | Abhikarta Table | Migration Notes |
|--------------|-----------------|-----------------|
| `agents` | `agents` | Add HITL, swarm fields; remove web_search fields (use external tools) |
| `agent_prompts` | (template system) | Convert to Abhikarta template library |
| `prompts` | (template system) | Map to agent/workflow templates |
| `prompt_revisions` | (version history) | May lose revision history or need custom migration |

#### 1.2 Conversation/Run Tables

| samjha Table | Abhikarta Table | Migration Notes |
|--------------|-----------------|-----------------|
| `conversations` | `executions` (partial) | Different concept - Abhikarta focuses on executions |
| `messages` | (execution logs) | Map to execution event logs |
| `runs` | `executions` | Core execution record |
| `run_events` | `execution_steps` | Step-by-step execution logs |
| `run_results` | (embedded in execution) | Flatten into execution record |
| `tool_traces` | `llm_logs` (partial) | Tool calls tracked differently |

#### 1.3 Bulk Doc Analysis Tables

| samjha Table | Abhikarta Table | Migration Notes |
|--------------|-----------------|-----------------|
| `sessions` | (custom) | May need custom table or adapt |
| `documents` | (custom) | Document processing not native to Abhikarta |
| `chains` | `workflows` | Similar concept |
| `chain_steps` | (workflow nodes) | Map to Abhikarta workflow node format |
| `chain_versions` | (workflow versions) | Version tracking |
| `bulk_doc_runs` | `executions` | Map to execution system |
| `step_results` | `execution_steps` | Execution step results |
| `workflows` | `workflows` | Direct mapping |
| `workflow_versions` | (version system) | Version tracking |
| `workflow_domains` | (custom metadata) | Domain scoping |
| `ingestion_profiles` | (custom) | Need custom implementation |
| `export_profiles` | (custom) | Need custom implementation |
| `execution_tasks` | (custom) | CSV task processing - custom |
| `job_queue_log` | (custom) | Job queue tracking |

#### 1.4 New Tables Required in Abhikarta

These tables exist in Abhikarta but not in samjha - need to configure:

| Table | Purpose | Action |
|-------|---------|--------|
| `users` | User management | Set up with existing users |
| `api_keys` | API authentication | Generate for existing integrations |
| `audit_logs` | Audit trail | Enable for compliance |
| `llm_providers` | LLM provider config | Configure Claude, OpenAI, etc. |
| `llm_models` | Model management | Register available models |
| `llm_model_permissions` | Access control | Set up permissions |
| `llm_logs` | LLM call logging | Enable for debugging |
| `mcp_servers` | MCP server registry | Register existing MCP servers |
| `mcp_tools` | MCP tool registry | Register MCP tools |
| `hitl_tasks` | Human review tasks | Configure if needed |
| `hitl_comments` | Review comments | Configure if needed |
| `hitl_assignments` | Task assignments | Configure if needed |
| `notification_channels` | Notifications | Set up email/webhook |
| `notification_templates` | Alert templates | Create templates |
| `settings` | System config | Configure defaults |

---

### 2. Tool Framework Migration

#### 2.1 Current Tool Inventory

```
external/tools/
├── parquet_agent/
│   ├── duck_db_tool.py         → Abhikarta FunctionTool
│   ├── list_tables_tool.py     → Abhikarta FunctionTool
│   ├── schema_tool.py          → Abhikarta FunctionTool
│   └── ...
└── duckdb_deep_diagnostics.py  → Abhikarta FunctionTool

tools/
└── tools_registry.py           → Merge into Abhikarta ToolsRegistry
```

#### 2.2 Tool Wrapper Template

```python
# Abhikarta tool wrapper pattern
from abhikarta.tools import BaseTool

class DuckDBQueryTool(BaseTool):
    name = "duckdb_query"
    description = "Execute SQL query on Parquet files"
    
    def __init__(self):
        # Import existing implementation
        from external.tools.parquet_agent.duck_db_tool import execute_query
        self._execute = execute_query
    
    def execute(self, sql: str, **params):
        return self._execute(sql, **params)
```

#### 2.3 MCP Tool Migration

Current MCP tools in `config/tools/` need registration in Abhikarta's MCP system:
- Register MCP servers in `mcp_servers` table
- Auto-discover tools via Abhikarta's MCP integration
- Map tool schemas to Abhikarta format

---

### 3. Agent Migration

#### 3.1 Agent Type Mapping

| samjha Agent Type | Abhikarta Equivalent | Notes |
|-------------------|---------------------|-------|
| `structured` (Parquet/SQL) | Custom data agent | Needs tool integration |
| `unstructured` | Document agent | Map to Abhikarta document processing |
| `external` (Web Research) | Custom web agent | Use HTTPTool base |

#### 3.2 Agent Definition Migration

Current agents in `external/config/agents/`:
- `ecommerce_agent.json`
- `sales.json`
- `widget_sales_agent.json`

Convert to Abhikarta agent template format:

```json
{
  "name": "ecommerce_agent",
  "template_type": "structured_data",
  "description": "...",
  "system_prompt": "...",
  "tools": ["duckdb_query", "list_tables", "get_schema"],
  "llm_config": {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096
  }
}
```

---

### 4. API Route Migration

#### 4.1 Route Mapping

| samjha Route | Abhikarta Route | Notes |
|--------------|-----------------|-------|
| `POST /api/auth/login` | `POST /api/auth/login` | Similar |
| `GET /api/agents` | `GET /api/agents` | Similar |
| `POST /api/agents` | `POST /api/agents` | Similar |
| `POST /api/agents/{id}/run` | `POST /api/agents/{id}/execute` | Rename |
| `GET /api/tools` | `GET /api/tools` | Similar |
| `POST /api/tools/{name}/execute` | `POST /api/tools/{name}/execute` | Similar |
| `GET /api/bulk-doc/*` | (custom blueprint) | Need to port |

#### 4.2 Breaking Changes

Existing clients using these endpoints need updates:
- Agent execution endpoint changes
- Response format differences
- Authentication flow changes (if any)

---

### 5. UI Migration

#### 5.1 Template Mapping

| samjha Template | Abhikarta Template | Action |
|-----------------|-------------------|--------|
| `home.html` | Dashboard | Adapt layout |
| `agent_chat.html` | Agent execution UI | Major rewrite |
| `admin.html` | Admin panel | Adapt to Abhikarta admin |
| `bulk_doc_analysis.html` | Custom | Port as blueprint |
| `tools_list.html` | Tools page | Minor changes |

#### 5.2 Static Assets

Migrate from `web/static/` to Abhikarta static folder:
- CSS files (may need theme adaptation)
- JavaScript files (update API calls)
- Images/icons

---

### 6. Feature Gap Analysis

#### 6.1 Features to Preserve (samjha → Abhikarta)

| Feature | Current Location | Migration Strategy |
|---------|-----------------|-------------------|
| Bulk Doc Analysis | `external/ai_bulk_doc_analysis/` | Port as Abhikarta blueprint |
| Vision Ingestion | `ingestion_service.py` | Integrate with Abhikarta |
| Web Research Agent | `web_research_agent.py` | Convert to Abhikarta agent |
| Parquet/DuckDB Querying | `parquet_agent/` | Convert to Abhikarta tools |
| Chain/Workflow Builder | `chain_validator.py`, etc. | Use Abhikarta workflow designer |
| Export Profiles | `export_service.py` | Port to Abhikarta |

#### 6.2 New Features from Abhikarta

| Feature | Value | Setup Required |
|---------|-------|---------------|
| HITL (Human-in-the-loop) | Human review gates | Configure workflows |
| Swarms | Multi-agent coordination | Define swarm agents |
| AI Organizations | Hierarchical agents | Design org structure |
| Visual Workflow Designer | No-code workflows | Train users |
| LLM Management | Provider/model config | Set up providers |
| Audit Logging | Compliance | Enable logging |
| API Key Management | Secure access | Generate keys |

---

### 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data loss during migration | Medium | High | Backup + rollback scripts |
| Feature regression | High | Medium | Comprehensive testing |
| API breaking changes | High | High | Compatibility shim |
| Performance degradation | Medium | Medium | Load testing |
| Learning curve | High | Low | Documentation + training |
| Vendor lock-in | High | High | Document exit strategy |

---

### 8. Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Pre-Migration | 1 week | None |
| Database Migration | 2 weeks | Pre-Migration |
| Tool Framework | 1-2 weeks | Database |
| Agent Migration | 1-2 weeks | Tools |
| API/Routes | 1-2 weeks | Agents |
| UI Migration | 2-3 weeks | API |
| New Features | 1-2 weeks | UI |
| Testing | 1-2 weeks | All above |
| Deployment | 1 week | Testing |
| Cleanup | 1 week | Deployment |

**Total Estimate: 10-16 weeks**

---

### 9. Success Criteria

- [ ] All existing agents functional in Abhikarta
- [ ] All existing workflows migrated and runnable
- [ ] No data loss in migration
- [ ] API backwards compatibility (or documented breaking changes)
- [ ] Performance meets or exceeds current baseline
- [ ] All tests passing
- [ ] Team trained on Abhikarta

---

### 10. Open Questions

1. **License Agreement** - Do we have permission to use/modify Abhikarta code?
2. **Support** - Who provides support for Abhikarta issues?
3. **Customization** - Can we add custom modules (like bulk doc analysis)?
4. **Banking Features** - Should we remove banking-specific code?
5. **Hosting** - Self-hosted or vendor-managed?

---

## Appendix A: File-by-File Migration Map

### Core Files

| samjha File | Action | Target |
|-------------|--------|--------|
| `web/app.py` | Replace | `abhikarta-web/app.py` |
| `core/auth_manager.py` | Merge | Abhikarta auth system |
| `core/mcp_handler.py` | Merge | Abhikarta MCP integration |
| `external/core/db/models.py` | Migrate | Abhikarta models |
| `external/agent/*.py` | Convert | Abhikarta agent framework |
| `external/routes/*.py` | Adapt | Abhikarta routes |
| `tools/tools_registry.py` | Merge | Abhikarta ToolsRegistry |

### Config Files

| samjha File | Action | Target |
|-------------|--------|--------|
| `config/users.json` | Migrate | Abhikarta `users` table |
| `external/config/agents/*.json` | Convert | Abhikarta agent templates |
| `external/config/domains/*.md` | Convert | Abhikarta system prompts |
| `external/config/prompts/*.md` | Convert | Abhikarta template library |
| `external/config/tools/*.json` | Register | Abhikarta tool registry |

---

## Appendix B: Quick Reference Commands

```bash
# Clone Abhikarta
git clone https://github.com/ajsinha/abhikarta-llm.git

# Run Abhikarta locally
cd abhikarta-llm
pip install -r requirements.txt
python run.py

# Access Abhikarta
# Web UI: http://localhost:5000
# Default login: admin / admin123

# Database migration (conceptual)
python scripts/migrate_samjha_to_abhikarta.py --source samjha.db --target abhikarta.db
```

---

*Document maintained by: [Your Team]*
*Next review: [Date]*

