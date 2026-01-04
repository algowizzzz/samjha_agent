# Prompts & APIs Status - Structured vs Web Research Agents

## Summary

**Total Prompts**: 12
- **Structured Agent**: 6 prompts
- **Web Research Agent**: 6 prompts

**APIs**: ✅ All core APIs completed

---

## Prompt Inventory

### Structured Data Agent (6 Prompts)

| # | Prompt File | High-Level Purpose | Used By | Status |
|---|-------------|-------------------|---------|--------|
| 1 | `decider.md` | **Gate/Decision Maker** - Analyzes user query, fills Query Spec, decides ASK_USER/EXECUTE/BLOCK. Outputs JSON contract (query_spec + query_spec_status + investigation_plan) | `decider.py` → `run_decider()` | ✅ Complete |
| 2 | `nl_to_sql_planner.md` | **SQL Generator** - Converts completed Query Spec into SQL. Must use exact table/view names from domain_md. Outputs SQL only (no prose). | `executor_nodes.py` → `sql_generation_node()` | ✅ Complete |
| 3 | `sql_plan_updater.md` | **SQL Plan Updater** - Updates SQL plan when spec changes or errors occur. Modifies existing SQL based on new requirements. | `executor_nodes.py` → `sql_generation_node()` (retry path) | ✅ Complete |
| 4 | `query_result_evaluator.md` | **Result Evaluator** - Evaluates SQL results against Query Spec. Checks output shape, grain, sanity checks. Outputs JSON evaluation. | `executor_nodes.py` → `evaluation_node()` | ✅ Complete |
| 5 | `ask_user_clarification.md` | **Clarification Generator** - Generates user-friendly clarification questions when Decider needs more info. Formats questions with context. | `decider.py` → `run_decider()` (when action=ASK_USER) | ✅ Complete |
| 6 | `response_commentary.md` | **Response Commentary** - Generates natural language explanation of SQL results. Converts tabular data into user-friendly narrative. | `executor_nodes.py` → `outcome_node()` | ✅ Complete |

### Web Research Agent (6 Prompts)

| # | Prompt File | High-Level Purpose | Used By | Status |
|---|-------------|-------------------|---------|--------|
| 1 | `web_research_decider.md` | **Gate/Decision Maker** - Analyzes research query, fills Research Spec, decides ASK_USER/EXECUTE/BLOCK. Outputs JSON contract (research_spec + research_spec_status + research_plan) | `web_research_decider.py` → `run_web_research_decider()` | ✅ Complete |
| 2 | `web_research_claim_extraction.md` | **Claim Extractor** - Extracts factual claims from web sources (Tavily results). Identifies key information, confidence levels, source attribution. | `web_research_executor_nodes.py` → `evidence_extraction_node()` | ✅ Complete |
| 3 | `web_research_conflict_detection.md` | **Conflict Detector** - Detects conflicting claims across sources. Identifies contradictions, severity levels, source disagreements. | `web_research_executor_nodes.py` → `conflict_detection_node()` | ✅ Complete |
| 4 | `web_research_synthesis.md` | **Final Synthesis** - Synthesizes final answer from EvidencePack. Resolves conflicts, provides citations, indicates confidence. Formats output per research_spec. | `web_research_synthesis.py` → `synthesize_final_answer()` | ✅ Complete |
| 5 | `web_research_ask_user_clarification.md` | **Clarification Generator** - Generates user-friendly clarification questions for research queries. Formats questions with search context. | `web_research_decider.py` → `run_web_research_decider()` (when action=ASK_USER) | ✅ Complete |
| 6 | `web_research_response_commentary.md` | **Response Commentary** - Generates natural language commentary on research findings. Formats evidence, conflicts, gaps into narrative. | `web_research_executor_nodes.py` → (optional commentary) | ✅ Complete |

---

## Prompt Comparison: Structured vs Web Research

### Decider Prompts

| Aspect | Structured (`decider.md`) | Web Research (`web_research_decider.md`) |
|--------|---------------------------|------------------------------------------|
| **Output Contract** | `query_spec` + `query_spec_status` | `research_spec` + `research_spec_status` |
| **Plan Type** | `investigation_plan` (tools: list_dir, inspect_table, search_glossary) | `research_plan` (Tavily search tool calls) |
| **Key Fields** | dimensions, metrics, filters, start_table, grain, time | user_question, intent_type, scope, quality_bar, constraints |
| **Domain Config** | `domain_md` (table schemas, metrics, business rules) | `domain_md` (authority domains, search scope, research depth) |
| **Status Tracking** | verified/inferred/missing/defaulted for each spec field | Same status tracking for research fields |

### Executor Prompts

| Structured Agent | Web Research Agent |
|------------------|-------------------|
| `nl_to_sql_planner.md` → Generates SQL | `web_research_claim_extraction.md` → Extracts claims |
| `sql_plan_updater.md` → Updates SQL | `web_research_conflict_detection.md` → Detects conflicts |
| `query_result_evaluator.md` → Evaluates SQL results | `web_research_synthesis.md` → Synthesizes final answer |
| `response_commentary.md` → Explains SQL results | `web_research_response_commentary.md` → Explains research |

### Clarification Prompts

| Structured Agent | Web Research Agent |
|------------------|-------------------|
| `ask_user_clarification.md` | `web_research_ask_user_clarification.md` |
| **Purpose**: Ask about missing metrics, dimensions, filters | **Purpose**: Ask about search scope, time range, authority domains |

---

## API Endpoints Status

### ✅ Core Agent APIs (Completed)

#### Agent Management APIs
- ✅ `GET /api/admin/agents` - List all agents
- ✅ `GET /api/admin/agents/<agent_id>` - Get agent details
- ✅ `POST /api/admin/agents` - Create new agent
- ✅ `PUT /api/admin/agents/<agent_id>` - Update agent
- ✅ `DELETE /api/admin/agents/<agent_id>` - Delete agent

#### Prompt Management APIs
- ✅ `GET /api/admin/prompts` - List prompts (with category filter)
- ✅ `GET /api/admin/prompts/<prompt_name>` - Get prompt content
- ✅ `POST /api/admin/prompts/<prompt_name>` - Save prompt content

#### Agent Run APIs (SSE Streaming)
- ✅ `POST /api/agents/<agent_id>/runs` - Start agent run
- ✅ `GET /api/runs/<run_id>/events` - SSE stream of events
- ✅ `GET /api/runs/<run_id>` - Get final run state (replay)
- ✅ `POST /api/runs/<run_id>/cancel` - Cancel running run

#### Conversation APIs
- ✅ `GET /api/agents/<agent_id>/conversations` - List conversations
- ✅ `GET /api/conversations/<conversation_id>/messages` - Get messages

### API Implementation Details

#### Agent Run Flow (Both Agent Types)

```python
# 1. Start Run
POST /api/agents/<agent_id>/runs
{
    "query": "user query",
    "conversation_id": "optional",
    "show_thinking": false,
    "model": "optional"
}
→ Returns: { "run_id": "...", "conversation_id": "..." }

# 2. Stream Events (SSE)
GET /api/runs/<run_id>/events
→ Streams: decider_output, executor_progress, final_result

# 3. Get Final State
GET /api/runs/<run_id>
→ Returns: Full run state with results
```

#### Agent Type Detection

The API automatically detects agent type:
- **Structured Agent**: Has `query_spec` in decider output
- **Web Research Agent**: Has `research_spec` in decider output

```python
# From agent_run_routes.py
if "research_spec" in decider_output:
    # Web research agent
    prior_research_spec = decider_output.get("research_spec")
else:
    # Structured agent
    prior_query_spec = decider_output.get("query_spec")
```

---

## Prompt Loading Mechanism

### File-Based (Default)
Prompts are stored in `external/config/prompts/*.md` and loaded via:
- `load_decider_prompt()` - Loads from file
- `load_prompt("prompt_name")` - Generic loader

### Database-Based (Override)
Prompts can be stored in DB and loaded per agent:
- `get_prompt_content(db, "prompt_name", category="structured|web_search")`
- Web research prompts support agent-specific overrides via `agent_id`

### Loading Priority
1. **Database** (if agent_id provided and prompt exists in DB)
2. **File** (fallback to `external/config/prompts/*.md`)

---

## Implementation Completeness

### ✅ Fully Implemented

1. **All 12 Prompts** - Complete and functional
2. **All Core APIs** - Agent management, runs, conversations, prompts
3. **SSE Streaming** - Real-time event streaming for agent runs
4. **State Persistence** - Runs, conversations, events stored in DB
5. **Prompt Management** - CRUD operations for prompts via API/UI
6. **Agent Type Detection** - Automatic detection of structured vs web research
7. **Continuity Packet** - Cross-turn state preservation for follow-up queries

### 🚧 Partially Implemented

1. **Prompt Versioning** - Prompts can be edited but no version history
2. **Prompt Testing** - No built-in prompt testing interface
3. **Prompt Analytics** - No metrics on prompt performance

### 📋 Not Implemented

1. **Prompt A/B Testing** - No support for testing prompt variants
2. **Prompt Templates** - No template system for prompt generation
3. **Prompt Validation** - No schema validation for prompt structure

---

## Key Differences: Structured vs Web Research

| Feature | Structured Agent | Web Research Agent |
|---------|------------------|-------------------|
| **Data Source** | Parquet/CSV files (DuckDB) | Web (Tavily API) |
| **Decider Output** | `query_spec` (SQL-focused) | `research_spec` (search-focused) |
| **Executor Tools** | list_dir, inspect_table, search_glossary, nl_to_sql_planner, execute_sql | Tavily search tools, claim extraction, conflict detection |
| **Final Output** | SQL query + tabular results | Synthesized answer + evidence pack |
| **Iterations** | Single execution (with retries) | Multiple iterations (deep research) |
| **State Accumulation** | Query spec preservation | Evidence pack accumulation |
| **Domain Config** | Table schemas, metrics, business rules | Authority domains, search scope, research depth |

---

## Usage Examples

### Structured Agent API

```python
# Start run
POST /api/agents/structured_agent_123/runs
{
    "query": "Show me total sales by country",
    "conversation_id": "conv_456"
}

# Stream events
GET /api/runs/run_789/events
# Streams: decider_output, executor_progress, final_result

# Get final state
GET /api/runs/run_789
# Returns: { "status": "SUCCESS", "final_sql": "...", "results": {...} }
```

### Web Research Agent API

```python
# Start run
POST /api/agents/web_agent_123/runs
{
    "query": "What are the latest AI trends?",
    "conversation_id": "conv_456"
}

# Stream events
GET /api/runs/run_789/events
# Streams: decider_output, research_progress, evidence_accumulation, final_answer

# Get final state
GET /api/runs/run_789
# Returns: { "status": "SUCCESS", "final_answer": "...", "evidence_pack": {...} }
```

---

## Conclusion

**Status**: ✅ **Production Ready**

- All 12 prompts implemented and functional
- All core APIs completed
- Both agent types fully operational
- SSE streaming for real-time updates
- State persistence and continuity support

**Next Steps** (Optional Enhancements):
- Prompt versioning and history
- Prompt testing interface
- Prompt performance analytics
- A/B testing for prompts

