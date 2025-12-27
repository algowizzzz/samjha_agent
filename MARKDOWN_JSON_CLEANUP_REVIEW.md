# Comprehensive Markdown & JSON Cleanup Review

## Executive Summary

This review identifies all markdown (.md) and JSON (.json) files that can be safely deleted, keeping only what's essential for the Parquet Agent.

**Files to KEEP (Essential):**
1. ✅ `domain_instructions/ecomm_domain.md` - Used by parquet agent
2. ✅ `external/config/prompts/decider.md` - Decider prompt
3. ✅ `external/config/tools/parquet_agent.json` - Main agent tool config
4. ✅ `external/schemas/*.json` (6 files) - Schema validation files
5. ✅ `config/tools/*.json` (9 parquet agent tool configs) - Active tool configs
6. ✅ `external/config/agent/queryagent_planner.json` - Used by QueryAgentConfig
7. ✅ Multi-5 query test script - `test_multi_followup` (script name to be confirmed)
8. ✅ `instructions/BUILD_PLAN.md` - Build instructions

---

## 1. ROOT DIRECTORY MARKDOWN/JSON FILES

### Files to DELETE:
- ❌ `UNUSED_CODE_ANALYSIS.md` - Analysis document (already created)
- ❌ `data_dictionary_risk_v3.json` - Not used by parquet agent

### Files to KEEP:
- ✅ `requirements.txt` - Python dependencies
- ✅ `instructions/` directory - Build plans and instructions

---

## 2. CONFIG DIRECTORY (`config/`)

### 2.1 `config/tools/` (172 JSON files)

**Files to KEEP (9 parquet agent tools):**
- ✅ `config/tools/list_dir.json`
- ✅ `config/tools/inspect_table.json`
- ✅ `config/tools/preview_rows.json`
- ✅ `config/tools/search_glossary.json`
- ✅ `config/tools/nl_to_sql_planner.json`
- ✅ `config/tools/sql_plan_updater.json`
- ✅ `config/tools/execute_sql.json`
- ✅ `config/tools/query_result_evaluator.json`
- ✅ `config/tools/query_safety_validator.json`

**Files to DELETE (163 files):**
- ❌ All other tool configs (Bank of Canada, ECB, FBI, Fed Reserve, Google Search, IMF, IR, Model Doc, SEC Edgar, Tavily, UN, World Bank, Wikipedia, Yahoo Finance, SQL Select, Web Crawler, etc.)
- ❌ `config/tools/ECB_REFACTORING_SUMMARY.md` - Documentation

**Note:** Tools registry loads all JSON files, but only parquet agent tools are actually used. Other tools will fail to load (expected warnings).

### 2.2 `config/agent/` (Directory)
- **Status**: Empty or contains unused files
- **Action**: Check contents, delete if empty/unused

### 2.3 `config/agent_welcome/` (Directory)
- **Status**: Not used by parquet agent
- **Action**: DELETE (only used by disabled doc_review routes)

### 2.4 `config/agent_welcome.md`
- **Status**: Not used
- **Action**: DELETE

### 2.5 `config/doc_review/`
- **Status**: Disabled product
- **Action**: DELETE entire directory
  - `config/doc_review/outline_templates/policy_template.json`

### 2.6 `config/ir/`
- **Status**: IR module deleted
- **Action**: DELETE entire directory
  - `config/ir/sp500_companies.json`

### 2.7 `config/prompts/doc-review/`
- **Status**: Disabled product
- **Action**: DELETE entire directory
  - `config/prompts/doc-review/content_improvement.txt`
  - `config/prompts/doc-review/gap_analysis.txt`

### 2.8 `config/literature/`
- **Status**: Not used
- **Action**: DELETE
  - `config/literature/wikipedia.txt`

### 2.9 Other config files:
- ❌ `config/application.properties` - Check if used
- ❌ `config/conetxt_risklimits` - Typo, likely unused
- ❌ `config/data_dictionary` - Directory, check if used
- ❌ `config/server.properties` - Check if used
- ❌ `config/users.json` - Check if used by auth

---

## 3. EXTERNAL CONFIG DIRECTORY (`external/config/`)

### 3.1 `external/config/agent/` (4 JSON files)

**Files to KEEP:**
- ✅ `external/config/agent/queryagent_planner.json` - Used by `QueryAgentConfig` class

**Files to DELETE:**
- ❌ `external/config/agent/deep_research_agent.json` - Deep research deleted
- ❌ `external/config/agent/doc_review_agent.json` - Doc review disabled
- ❌ `external/config/agent/model_doc_agent.json` - Model doc disabled

### 3.2 `external/config/agent_welcome/`
- **Status**: Previously used by `agent_routes.py` for `/api/agent/welcome-message`
- **Decision**: ❌ **DELETE** (feature not needed)
- **Implementation**: ✅ Deleted folder + removed the endpoint + removed UI references in `web/templates/agent_chat.html`

### 3.3 `external/config/data_analysis_agents/`
- **Status**: Previously used by `agent_routes.py` for `/api/agent/data-analysis-agents`
- **Decision**: ❌ **DELETE** (feature not needed)
- **Implementation**: ✅ Deleted folder + removed the endpoint(s) + removed UI references in `web/templates/agent_chat.html`

### 3.4 `external/config/data_dictionary/`
- **Status**: Previously used by `agent_routes.py` for `/api/agent/config-files`
- **Decision**: ❌ **DELETE** (feature not needed)
- **Implementation**: ✅ Deleted folder + removed the endpoint + removed UI references in `web/templates/agent_chat.html`

### 3.5 `external/config/prompts/`
- **Files to KEEP:**
  - ✅ `external/config/prompts/decider.md` - **ESSENTIAL** - Decider prompt

### 3.6 `external/config/tools/` (18 JSON files)

**Files to KEEP:**
- ✅ `external/config/tools/parquet_agent.json` - Main agent tool config

**Files to DELETE (17 files):**
- ❌ All doc_processing tool configs (analyze_heading_structure, annotate_markdown_for_ui, apply_changes_deterministic, assemble_improved_markdown, build_file_metadata, build_index, chunk_markdown, compute_file_stats, convert_to_markdown, decide_chunking_strategy, detect_file_type, extract_images, extract_section_by_headings, extract_section_by_toc, generate_toc_from_index, load_outline_template)
- ❌ `external/config/tools/deep_research_agent.json` - Deep research deleted

---

## 4. EXTERNAL DATA DIRECTORY (`external/data/`)

### 4.1 `external/data/doc_review/` (ENTIRE DIRECTORY)
- **Status**: Disabled product data
- **Action**: ❌ **DELETE ENTIRE DIRECTORY**
  - `external/data/doc_review/improved/` - 107 markdown files
  - `external/data/doc_review/markdown/` - 50 markdown files
  - `external/data/doc_review/state/` - 5 JSON files
  - `external/data/doc_review/uploads/` - 91 files (PDFs, MD, TXT)
  - `external/data/doc_review/images/` - 12 PNG files

**Total**: ~265 files, ~100MB+ of unused data

---

## 5. EXTERNAL SCHEMAS DIRECTORY (`external/schemas/`)

### Files to KEEP (ALL 6 files - ESSENTIAL):
- ✅ `external/schemas/decider_output.schema.json` - Used by schema validators
- ✅ `external/schemas/executor_report.schema.json` - Used by schema validators
- ✅ `external/schemas/investigation_plan_step.schema.json` - Used by schema validators
- ✅ `external/schemas/policy_limits.schema.json` - Used by schema validators
- ✅ `external/schemas/query_spec.schema.json` - Used by schema validators
- ✅ `external/schemas/query_spec_status.schema.json` - Used by schema validators

**Action**: ✅ **KEEP ALL** - All are actively used for validation

---

## 6. DOCS DIRECTORY (`docs/`)

### 6.1 `docs/userguides/` (29 markdown files)

**Files to KEEP (Parquet Agent related):**
- ✅ `docs/userguides/Backend_Parquet_Agent_Technical_Documentation.md` - Technical docs
- ✅ `docs/userguides/Creating_Domain_Configuration_Guide.md` - Domain config guide
- ✅ `docs/userguides/Domain_Configuration_Fillable_Template.md` - Domain template
- ✅ `docs/userguides/QUERY_SPEC_INCOMPLETE_Error_Explanation.md` - Error docs
- ✅ `docs/FRONTEND_DEBUGGING_GUIDE.md` - Frontend debugging

**Files to DELETE (24 files - Tool reference guides for unused tools):**
- ❌ `docs/userguides/API Client README.md` - Not parquet agent
- ❌ `docs/userguides/Bank_of_Canada_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/CHINA_PEOPLES_BANK_DOCUMENTATION.md` - Unused tool
- ❌ `docs/userguides/DuckDB_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/Enhanced EDGAR MCP Tools - Reference Guide.md` - Unused tool
- ❌ `docs/userguides/European_Central_Bank_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/FBI_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/Federal_Reserve_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/FRANCE_BANQUE_DE_FRANCE_DOCUMENTATION.md` - Unused tool
- ❌ `docs/userguides/Google_Search_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/IMF_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/IMPLEMENTATION_SUMMARY.md` - Historical
- ❌ `docs/userguides/INDIA_RESERVE_BANK_DOCUMENTATION.md` - Unused tool
- ❌ `docs/userguides/Investor_Relations_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/JAPAN_BANK_OF_JAPAN_DOCUMENTATION.md` - Unused tool
- ❌ `docs/userguides/MSDOC_Search_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/SEC_Edgar_Search_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/SQL_Select_Search_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/Tavily_Search_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/United_Nations_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/Web_Crawler_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/Wikipedia_Search_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/World_Bank_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/Yahoo_Finance_MCP_Tool_Reference_Guide.md` - Unused tool
- ❌ `docs/userguides/deep_research_agent.md` - Deep research deleted

### 6.2 `docs/requirements/`
- **Status**: Check if needed
- **Action**: Review `docs/requirements/sajhamcpserver_requirements.docx`

---

## 7. INSTRUCTIONS DIRECTORY (`instructions/`)

### Files to KEEP (ALL):
- ✅ `instructions/BUILD_PLAN.md` - Build instructions
- ✅ `instructions/controller_loop_pseudocode.md` - Implementation docs
- ✅ `instructions/instructions_plan.md` - Planning docs
- ✅ `instructions/mcp_tools_implementation.md` - Tools implementation

**Action**: ✅ **KEEP ALL** - All are relevant to parquet agent implementation

---

## 8. DATA DIRECTORY (`data/`)

### 8.1 `data/agent_state/` (28 JSON files)
- **Status**: Session state files (runtime data)
- **Action**: ⚠️ **KEEP** - Runtime session data (may be cleaned periodically)

### 8.2 `data/docreview/`
- **Status**: Disabled product
- **Action**: ❌ **DELETE** - 5 files (PDFs, policy template)

### 8.3 `data/templates/`
- **Status**: Check if used
- **Action**: Review `data/templates/README.md` and `policy_template.md`

### 8.4 Other data directories:
- ⚠️ `data/duckdb/` - Check if used by parquet agent
- ⚠️ `data/msdocs/` - Check if used
- ⚠️ `data/sqlselect/` - Check if used

---

## 9. EXTERNAL MIGRATION SUMMARY

### Files to DELETE:
- ❌ `external/MIGRATION_SUMMARY.md` - Historical migration doc

---

## 10. TEST FILES

### Files to KEEP:
- ✅ All test files in `tests/agent/` (9 files) - Active tests
- ✅ `tests/tools/test_parquet_agent_tools.py` - Tool tests
- ✅ **Multi-5 query test script** (if found / restored)

**Note:** As of the last scan, the source script is not present; only a compiled cache file exists:
- `__pycache__/test_multi_followup.cpython-313-pytest-9.0.2.pyc`
So if you want to keep and run it, we will need to restore/recreate `test_multi_followup.py`.

### Files to DELETE:
- ❌ Any test files in root directory (already deleted)

---

## 11. WEB VSCODE EXTENSIONS

### `web/vscode_extensions/doc-review-vfs/`
- **Status**: Doc review extension (disabled product)
- **Action**: ❌ **DELETE ENTIRE DIRECTORY** - Includes node_modules, package.json, tsconfig.json, README.md

---

## 12. MCP TOOLS CLEANUP

### Current State:
- **Total tool implementations**: 36 files in `tools/impl/`
- **Total tool configs**: 172 JSON files in `config/tools/`
- **Parquet agent tools**: 9 tools in `external/tools/parquet_agent/` (ACTIVE)
- **Tavily tools**: 4 tools (KEEP all 4)

### Tools to KEEP:

#### Parquet Agent Tools (9 tools - ESSENTIAL):
- ✅ `external/tools/parquet_agent/list_dir.py` + `config/tools/list_dir.json`
- ✅ `external/tools/parquet_agent/inspect_table.py` + `config/tools/inspect_table.json`
- ✅ `external/tools/parquet_agent/preview_rows.py` + `config/tools/preview_rows.json`
- ✅ `external/tools/parquet_agent/search_glossary.py` + `config/tools/search_glossary.json`
- ✅ `external/tools/parquet_agent/nl_to_sql_planner.py` + `config/tools/nl_to_sql_planner.json`
- ✅ `external/tools/parquet_agent/sql_plan_updater.py` + `config/tools/sql_plan_updater.json`
- ✅ `external/tools/parquet_agent/execute_sql.py` + `config/tools/execute_sql.json`
- ✅ `external/tools/parquet_agent/query_result_evaluator.py` + `config/tools/query_result_evaluator.json`
- ✅ `external/tools/parquet_agent/query_safety_validator.py` + `config/tools/query_safety_validator.json`

#### Tavily Tools (KEEP all 4):
- ✅ `tools/impl/tavily_tool_refactored.py` (implementation - contains all 4 classes)
- ✅ `config/tools/tavily_web_search.json`
- ✅ `config/tools/tavily_research_search.json`
- ✅ `config/tools/tavily_news_search.json`
- ✅ `config/tools/tavily_domain_search.json`

### Tools to DELETE:

#### Tool Implementations (`tools/impl/` - 33 files):
- ❌ `bank_of_canada_tool_refactored.py`
- ❌ `china_central_bank.py`
- ❌ `create_column_embeddings_tool.py`
- ❌ `duckdb_olap_tools_refactored.py`
- ❌ `enhanced_edgar_tool.py`
- ❌ `european_central_bank_tool_refactored.py`
- ❌ `fbi_tool_refactored.py`
- ❌ `fed_reserve_tool_refactored.py`
- ❌ `france_central_bank.py`
- ❌ `google_search_tool_refactored.py`
- ❌ `imf_tool_refactored.py`
- ❌ `india_central_bank.py`
- ❌ `investor_relations_tool_refactored.py`
- ❌ `japan_central_bank.py`
- ❌ `llm_map_tool.py`
- ❌ `model_doc_assemble_doc_tool.py`
- ❌ `model_doc_build_hierarchy_tool.py`
- ❌ `model_doc_compute_stats_tool.py`
- ❌ `model_doc_list_files_tool.py`
- ❌ `model_doc_load_template_tool.py`
- ❌ `model_doc_parse_ast_tool.py`
- ❌ `model_doc_read_file_tool.py`
- ❌ `model_doc_save_doc_tool.py`
- ❌ `msdoc_tools_tool_refactored.py`
- ❌ `pattern_match_tool.py`
- ❌ `search_entity_in_data_tool.py`
- ❌ `sec_edgar_tool_refactored.py`
- ❌ `sqlselect_tool_refactored.py`
- ❌ `united_nations_tool_refactored.py`
- ❌ `vector_search_tool.py`
- ❌ `webcrawler_tool_refactored.py`
- ❌ `wikipedia_tool.py`
- ❌ `world_bank_tool.py`
- ❌ `yahoo_finance_tool.py`

#### Tool Configs (`config/tools/` - 163 files):
- ❌ All tool configs except:
  - 9 parquet agent tool configs (KEEP)
  - 4 Tavily tool configs (KEEP)
  - Total to DELETE: ~159 config files

**Note**: Tools registry will show warnings for deleted tools, but parquet agent will work fine.

### Summary:
- **Keep**: 9 parquet agent tools + 4 Tavily tools = 13 tools total
- **Delete**: ~33 tool implementations + ~160 tool configs = ~193 files
- **Impact**: Tools registry will have fewer tools, but parquet agent functionality unaffected

---

## 13. SUMMARY OF DELETIONS

### High Priority (Definitely Unused):
1. ❌ `external/data/doc_review/` - **ENTIRE DIRECTORY** (~265 files, ~100MB)
2. ❌ `config/tools/*.json` - **163 unused tool configs** (keep only 9 parquet tools)
3. ❌ `external/config/tools/*.json` - **17 doc_processing configs** (keep only parquet_agent.json)
4. ❌ `external/config/agent/*.json` - **3 disabled product configs** (keep only queryagent_planner.json)
5. ❌ `docs/userguides/*.md` - **24 tool reference guides** (keep only 5 parquet agent docs)
6. ❌ `config/doc_review/` - **ENTIRE DIRECTORY**
7. ❌ `config/ir/` - **ENTIRE DIRECTORY**
8. ❌ `config/prompts/doc-review/` - **ENTIRE DIRECTORY**
9. ❌ `config/literature/` - **ENTIRE DIRECTORY**
10. ❌ `web/vscode_extensions/doc-review-vfs/` - **ENTIRE DIRECTORY**
11. ❌ `data/docreview/` - **ENTIRE DIRECTORY**
12. ❌ `external/MIGRATION_SUMMARY.md` - Historical doc
13. ❌ `UNUSED_CODE_ANALYSIS.md` - Analysis doc (already created)
14. ❌ `data_dictionary_risk_v3.json` - Root level, not used

### Medium Priority (DELETE - User Requested):
15. ❌ `external/config/agent_welcome/` - Used by frontend UI dropdowns (DELETE + remove UI code)
16. ❌ `external/config/data_analysis_agents/` - Used by frontend UI for agent management (DELETE + remove UI code)
17. ❌ `external/config/data_dictionary/` - Used by frontend UI dropdowns (DELETE + remove UI code)

**Note:** These require removing UI code in `agent_chat.html` that references these configs (lines 623-634, 1181, 1203, 1218, 1233, 1510, 1573, 1645, 1675) and removing routes in `agent_routes.py` (lines 60-86, 88-181, 183-262).

### Runtime Data (Keep - Auto-generated):
18. ⚠️ `data/agent_state/*.json` - **RUNTIME SESSION DATA** - Created automatically by `AgentStateManager` when agent runs. These are session persistence files for follow-up queries. Current implementation uses in-memory sessions, but these files may be used in future. **KEEP** - Can be cleaned periodically but don't delete the directory.
20. ⚠️ `data/templates/` - Review if used
21. ⚠️ `data/duckdb/` - Review if used
22. ⚠️ `data/msdocs/` - Review if used
23. ⚠️ `data/sqlselect/` - Review if used
24. ⚠️ `config/application.properties` - Review if used
25. ⚠️ `config/server.properties` - Review if used
26. ⚠️ `config/users.json` - Review if used by auth

### Files to KEEP (Essential):
- ✅ `domain_instructions/ecomm_domain.md`
- ✅ `external/config/prompts/decider.md`
- ✅ `external/config/tools/parquet_agent.json`
- ✅ `external/config/agent/queryagent_planner.json`
- ✅ `external/schemas/*.json` (6 files)
- ✅ `config/tools/*.json` (9 parquet agent tool configs)
- ✅ `instructions/*.md` (4 files)
- ✅ `docs/userguides/Backend_Parquet_Agent_Technical_Documentation.md`
- ✅ `docs/userguides/Creating_Domain_Configuration_Guide.md`
- ✅ `docs/userguides/Domain_Configuration_Fillable_Template.md`
- ✅ `docs/userguides/QUERY_SPEC_INCOMPLETE_Error_Explanation.md`
- ✅ `docs/FRONTEND_DEBUGGING_GUIDE.md`
- ✅ Multi-5 query test script (to be identified)

---

## 14. ESTIMATED CLEANUP IMPACT

**Files to Delete:**
- ~265 files in `external/data/doc_review/`
- ~159 tool configs in `config/tools/` (keep 9 parquet + 4 Tavily = 13 total)
- ~33 tool implementations in `tools/impl/` (keep only tavily_tool_refactored.py)
- ~17 tool configs in `external/config/tools/`
- ~24 documentation files in `docs/userguides/`
- ~10+ directories and misc files

**Total**: ~490+ files, ~100MB+ disk space

**Disk Space Savings**: ~100MB+ (mostly from doc_review data)

---

## 15. VERIFICATION CHECKLIST

After cleanup, verify:
- [ ] Parquet agent still loads domain files (`domain_instructions/ecomm_domain.md`)
- [ ] Decider prompt loads (`external/config/prompts/decider.md`)
- [ ] All 9 parquet agent tools load from `config/tools/`
- [ ] 4 Tavily tools load correctly
- [ ] Tools registry doesn't break (warnings for deleted tools are expected)
- [ ] Schema validators work (6 schema JSON files)
- [ ] QueryAgentConfig loads (`external/config/agent/queryagent_planner.json`)
- [ ] Multi-5 query test script still works

---

## 16. CORE MCP SERVER - REQUIRED

### Essential Core Components (KEEP ALL):
- ✅ `core/auth_manager.py` - **REQUIRED** - Authentication for login/UI
- ✅ `core/mcp_handler.py` - **REQUIRED** - MCP protocol handler for `/api/tools/execute`
- ✅ `tools/tools_registry.py` - **REQUIRED** - Tool loading and execution
- ✅ `routes/api_routes.py` - **REQUIRED** - API endpoints including `/api/tools/execute`
- ✅ `routes/auth_routes.py` - **REQUIRED** - Login/logout routes

**Why needed:**
- `web/app.py` imports and uses all of these
- `/api/tools/execute` endpoint requires `mcp_handler` and `tools_registry`
- UI login requires `auth_manager`
- Parquet agent tool execution goes through MCP protocol

**Action**: ✅ **KEEP ALL** - These are essential infrastructure, not optional.

---

## 17. DATA/AGENT_STATE EXPLANATION

**What is `data/agent_state/*.json`?**
- **Purpose**: Session persistence files created by `AgentStateManager` class
- **Usage**: Stores full agent state per session for follow-up query continuity
- **Current Status**: 
  - `AgentStateManager` exists in `external/agent/state_manager.py`
  - Current implementation (`base_agent.py`) uses **in-memory sessions** (`self._sessions`)
  - JSON files are created if `AgentStateManager` is used (not currently active)
- **Action**: ⚠️ **KEEP DIRECTORY** - Files are runtime-generated, can be cleaned periodically but directory structure is needed

---

**Generated**: 2025-01-XX
**Review Scope**: Markdown and JSON files for Parquet Agent v1
**Total Files Reviewed**: ~500+ markdown/JSON files

