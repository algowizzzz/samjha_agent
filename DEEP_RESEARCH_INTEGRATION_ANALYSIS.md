# Open Deep Research Integration Analysis

## Executive Summary

This document analyzes integrating LangChain's [Open Deep Research](https://github.com/langchain-ai/open_deep_research) as a new agent type in the existing system. The system currently supports:
- **Structured Agents** (`agent_type="structured"`): SQL/Parquet data analysis
- **External Agents** (`agent_type="external"`): Web search research using Tavily

**Proposed:** Add **Deep Research Agents** (`agent_type="deep_research"`) based on Open Deep Research architecture.

---

## 1. Current Architecture Analysis

### 1.1 Agent Type System

**Database Model** (`external/core/db/models.py`):
```python
agent_type: Mapped[str] = mapped_column(String(32), nullable=False)  
# Current values: "structured" | "external"
```

**Routing Logic** (`external/routes/agent_run_routes.py`):
- Lines 578-614: Routes based on `agent_type`
- `agent_type == "external"` → `handle_web_research_query()`
- `else` → `handle_query()` (structured agent)

**Agent Handlers**:
- Structured: `external/agent/parquet_agent.py` → `handle_query()`
- External: `external/agent/web_research_agent.py` → `handle_web_research_query()`

### 1.2 Current Agent Types

#### Structured Agents
- **Purpose**: SQL generation and execution on Parquet data
- **Architecture**: Controller → Decider → Executor (LangGraph)
- **State**: `ControllerState` / `ExecutorState`
- **Tools**: SQL generation, execution, validation
- **Config**: `domain_md`, `data_folder`, `model`

#### External Agents (Web Search)
- **Purpose**: Deep web research with domain restrictions
- **Architecture**: ResearchController → Decider → Executor (iterative)
- **State**: `ResearchControllerState` / `ResearchExecutorState`
- **Tools**: Tavily search, claim extraction, conflict detection
- **Config**: `domain_md`, `tavily_api_key`, `search_scope_*`, `default_research_depth`

---

## 2. Open Deep Research Architecture

### 2.1 Core Components

Based on the [repository](https://github.com/langchain-ai/open_deep_research):

**Architecture Flow**:
1. **Summarization** → Summarizes search API results
2. **Research** → Powers the search agent (main LLM)
3. **Compression** → Compresses research findings
4. **Final Report** → Writes the final report

**Key Features**:
- LangGraph-based state machine
- Multi-model support (OpenAI, Anthropic, OpenRouter, Ollama)
- Multiple search APIs (Tavily, MCP, native web search)
- Configurable research depth and quality
- Evaluation framework (Deep Research Bench)

### 2.2 Configuration Structure

From `configuration.py` (inferred from README):
- **Models**: `summarization_model`, `research_model`, `compression_model`, `final_report_model`
- **Search API**: `search_api` (Tavily, MCP, native)
- **MCP Config**: `mcp_config` for Model Context Protocol
- **Research Settings**: Depth, quality thresholds, max iterations

### 2.3 State Management

Uses LangGraph `StateGraph` with:
- Research state (queries, sources, findings)
- Iteration tracking
- Quality metrics
- Final report generation

---

## 3. Integration Requirements

### 3.1 Database Schema Changes

**Agent Model** (`external/core/db/models.py`):

**Option A: Extend existing fields** (Recommended)
```python
# Add deep research specific fields (similar to web search fields)
deep_research_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
# Contains: models, search_api, mcp_config, research_settings
```

**Option B: Separate table** (More normalized)
```python
class DeepResearchConfig(Base):
    agent_id: Mapped[str] = ForeignKey("agents.id")
    summarization_model: Mapped[str]
    research_model: Mapped[str]
    compression_model: Mapped[str]
    final_report_model: Mapped[str]
    search_api: Mapped[str]  # "tavily" | "mcp" | "native"
    mcp_config: Mapped[Optional[dict]] = mapped_column(JSON)
    research_settings: Mapped[dict] = mapped_column(JSON)
```

**Recommendation**: Option A (simpler, consistent with current pattern)

### 3.2 New Agent Type Handler

**File**: `external/agent/deep_research_agent.py`

**Structure** (mirroring `web_research_agent.py`):
```python
def handle_deep_research_query(
    user_query: str,
    conversation_history: list,
    prior_state: Optional[DeepResearchControllerState] = None,
    tools_registry: Optional[ToolsRegistry] = None,
    policy_limits: Optional[dict] = None,
    show_thinking: bool = False,
    agent_id: Optional[str] = None,
    on_decider_output: Optional[Callable] = None,
    continuity_packet: Optional[dict] = None,
) -> dict:
    """
    Controller orchestrates:
      Summarization → Research → Compression → Final Report
    """
    # Initialize state
    # Load Open Deep Research graph
    # Execute research pipeline
    # Return formatted result
```

### 3.3 State Types

**File**: `external/agent/state_types.py`

Add new state classes:
```python
class DeepResearchControllerState(TypedDict, total=False):
    user_query: str
    conversation_history: Optional[List[Dict[str, str]]]
    research_config: Dict[str, Any]  # From agent.deep_research_config
    research_state: Dict[str, Any]  # Open Deep Research internal state
    iteration_count: int
    show_thinking: bool
    agent_id: Optional[str]
    # ... similar to ResearchControllerState
```

### 3.4 Routing Updates

**File**: `external/routes/agent_run_routes.py`

**Line 578-614**: Add new condition:
```python
agent_type = agent.get("agent_type") if agent else "structured"

if agent_type == "external":
    # Web research agent (existing)
    from external.agent.web_research_agent import handle_web_research_query
    result = handle_web_research_query(...)
elif agent_type == "deep_research":
    # Deep research agent (NEW)
    from external.agent.deep_research_agent import handle_deep_research_query
    result = handle_deep_research_query(...)
else:
    # Structured agent (default)
    from external.agent.parquet_agent import handle_query
    result = handle_query(...)
```

**Line 627-646**: Add result handling:
```python
if status == "SUCCESS":
    if agent_type == "external":
        # Web research agent response format
        ...
    elif agent_type == "deep_research":
        # Deep research agent response format
        final_report = result.get("final_report") or result.get("result_summary", "")
        research_sources = result.get("sources", [])
        # Save to DB similar to web research
        ...
    else:
        # Structured agent response format
        ...
```

### 3.5 Open Deep Research Integration

**Approach Options**:

#### Option A: Direct Integration (Recommended)
- Clone/copy Open Deep Research source into `external/agent/deep_research/`
- Adapt their LangGraph implementation to our state management
- Use their configuration system with our DB storage

**Pros**: Full control, can customize
**Cons**: More code to maintain, need to keep up with updates

#### Option B: Library Import
- Install `open-deep-research` as a package (if available)
- Wrap their API in our handler

**Pros**: Less code, automatic updates
**Cons**: May not be available as package, less control

#### Option C: Hybrid
- Use their core research logic
- Wrap with our state management and routing
- Adapt their configuration to our DB schema

**Recommendation**: Option C (best balance)

### 3.6 Tool Integration

**Search APIs**:
- **Tavily**: ✅ Already integrated (`tools/impl/tavily_tool_refactored.py`)
- **MCP**: ✅ Already integrated (`core/mcp_handler.py`, `tools/base_mcp_tool.py`)
- **Native**: Need to implement (Anthropic/OpenAI native web search APIs)

**Tools Registry** (`tools/tools_registry.py`):
- Can reuse existing Tavily tool
- Can leverage existing MCP handler for MCP-based search
- May need to add native search tools for Anthropic/OpenAI
- Deep research may need additional tools (summarization, compression helpers)

### 3.7 UI Updates

**Admin Panel** (`external/web/templates/admin.html`):

1. **Agent Creation Form**:
   - Add "Deep Research" option to agent type dropdown
   - Add configuration fields:
     - Models (summarization, research, compression, final report)
     - Search API selection
     - MCP configuration
     - Research settings (depth, quality thresholds)

2. **Agent Management**:
   - Filter by agent type (add "deep_research")
   - Display deep research specific config

3. **System Prompts** (if needed):
   - Add category: `"deep_research"`
   - Or reuse `"web_search"` category

**Chat Interface** (`external/web/templates/agent_chat.html`):
- Should work automatically (uses same SSE event stream)
- May need custom result rendering for final report format

### 3.8 Configuration Management

**Agent Persistence** (`external/agent/persistence.py`):

Update `get_agent_db()`:
```python
def get_agent_db(db, agent_id: str) -> Optional[Dict[str, Any]]:
    # ... existing code ...
    if hasattr(a, 'deep_research_config'):
        result["deep_research_config"] = a.deep_research_config
    return result
```

Update `create_agent()` / `update_agent()`:
```python
# Handle deep_research_config JSON field
if agent_type == "deep_research":
    # Validate and store deep_research_config
    ...
```

---

## 4. Implementation Plan

### Phase 1: Foundation (Week 1)
1. ✅ Database schema update (add `deep_research_config` JSON field)
2. ✅ Create `DeepResearchControllerState` and related state types
3. ✅ Add routing logic in `agent_run_routes.py`
4. ✅ Create `deep_research_agent.py` skeleton

### Phase 2: Core Integration (Week 2)
1. ✅ Integrate Open Deep Research source code
2. ✅ Adapt LangGraph implementation to our state management
3. ✅ Implement `handle_deep_research_query()`
4. ✅ Connect to tools registry (Tavily, MCP, native search)

### Phase 3: Configuration & UI (Week 3)
1. ✅ Update admin UI for deep research agent creation
2. ✅ Add configuration form fields
3. ✅ Update persistence layer
4. ✅ Test agent creation and configuration

### Phase 4: Testing & Refinement (Week 4)
1. ✅ End-to-end testing
2. ✅ Error handling and edge cases
3. ✅ Performance optimization
4. ✅ Documentation

---

## 5. Key Differences: Deep Research vs. Web Search

| Aspect | Web Search (External) | Deep Research |
|--------|----------------------|---------------|
| **Purpose** | Domain-restricted research | Comprehensive deep research |
| **Architecture** | Decider → Executor (iterative) | Summarization → Research → Compression → Report |
| **Search** | Tavily only (domain filtered) | Multiple APIs (Tavily, MCP, native) |
| **Output** | Evidence pack with sources | Final report with citations |
| **Iterations** | 2-4 iterations (configurable) | Multi-stage pipeline |
| **Models** | Single model | 4 models (summarization, research, compression, report) |
| **Quality** | Conflict detection, claim extraction | Compression, quality scoring |

---

## 6. Technical Considerations

### 6.1 Model Configuration

**Challenge**: Deep Research uses 4 different models
**Solution**: Store in `deep_research_config` JSON:
```json
{
  "summarization_model": "openai:gpt-4.1-mini",
  "research_model": "openai:gpt-4.1",
  "compression_model": "openai:gpt-4.1",
  "final_report_model": "openai:gpt-4.1"
}
```

### 6.2 Search API Selection

**Challenge**: Multiple search APIs (Tavily, MCP, native)
**Solution**: 
- Reuse existing Tavily integration
- Integrate MCP handler (check if exists)
- Add native search for Anthropic/OpenAI

### 6.3 State Management

**Challenge**: Open Deep Research has its own state structure
**Solution**: 
- Wrap their state in our `DeepResearchControllerState`
- Map between their internal state and our controller state
- Use LangGraph's state management within our handler

### 6.4 Cost Management

**Challenge**: 4 models + multiple search APIs = higher cost
**Solution**:
- Add cost tracking in state
- Set policy limits (max API calls, max tokens)
- Allow users to configure model selection (cheaper models for summarization)

---

## 7. Migration Path

### For Existing Users:
- No breaking changes (new agent type, not modifying existing)
- Existing structured/external agents continue to work

### For New Deep Research Agents:
1. Create agent with `agent_type="deep_research"`
2. Configure models and search APIs
3. Set research settings (depth, quality)
4. Start using immediately

---

## 8. Open Questions

1. **Dependencies**: Does Open Deep Research have specific Python dependencies we need to add?
   - **Answer**: Need to check their `pyproject.toml` / `requirements.txt` when cloning repo
   
2. **MCP Integration**: Do we already have MCP handler, or need to build it?
   - **Answer**: ✅ **YES** - We have `core/mcp_handler.py` and `tools/base_mcp_tool.py`
   - MCP handler is initialized in `web/app.py` and passed to routes
   - Can reuse existing MCP infrastructure
   
3. **Native Search**: How to implement native web search for Anthropic/OpenAI?
   - **Answer**: Need to check if Open Deep Research has native search implementation
   - May need to create wrapper tools for Anthropic/OpenAI native search APIs
   
4. **Evaluation**: Should we integrate Deep Research Bench evaluation?
   - **Answer**: Optional - can add later if needed for benchmarking
   
5. **Prompt System**: Do we need separate prompts for deep research, or reuse web_search category?
   - **Answer**: Likely need new category `"deep_research"` or separate prompts
   - Deep research has different stages (summarization, research, compression, report)
   
6. **Domain Restrictions**: Should deep research support domain restrictions like web search?
   - **Answer**: Open Deep Research is designed for comprehensive research (no domain restrictions)
   - But we could add optional domain filtering if needed

---

## 9. Recommendations

### Immediate Actions:
1. ✅ Review Open Deep Research source code structure
2. ✅ Identify core components to integrate
3. ✅ Plan database schema changes
4. ✅ Design state management approach

### Architecture Decisions:
1. **Use Option A** for database schema (JSON field, consistent with current pattern)
2. **Use Option C** for integration (hybrid approach)
3. **Reuse existing tools** where possible (Tavily, tools registry)
4. **Separate prompts category** if needed, or reuse `web_search`

### Risk Mitigation:
1. Start with minimal viable integration
2. Test with single model first (research_model only)
3. Gradually add other models (summarization, compression, report)
4. Monitor costs and performance

---

## 10. Next Steps

1. **Review this analysis** with team
2. **Clone Open Deep Research** repository locally
3. **Examine source code** structure (`src/` folder)
4. **Identify integration points** (LangGraph nodes, state, config)
5. **Create proof of concept** (basic handler + routing)
6. **Iterate** based on testing

---

## Appendix: File Structure

```
external/agent/
├── parquet_agent.py          # Structured agents
├── web_research_agent.py      # External agents (web search)
├── deep_research_agent.py     # NEW: Deep research agents
├── state_types.py             # Add DeepResearchControllerState
└── deep_research/             # NEW: Open Deep Research source
    ├── graph.py               # LangGraph implementation
    ├── nodes.py               # Research nodes
    ├── state.py               # Internal state
    └── config.py              # Configuration

external/routes/
└── agent_run_routes.py        # Add routing for deep_research

external/core/db/
└── models.py                   # Add deep_research_config field

external/web/templates/
└── admin.html                  # Add deep research UI
```

---

**Status**: Analysis Complete - Ready for Implementation Planning

---

## 11. Summary & Action Items

### Key Findings

✅ **Existing Infrastructure**:
- MCP handler already exists (`core/mcp_handler.py`)
- Tavily tool already integrated (`tools/impl/tavily_tool_refactored.py`)
- Tools registry supports dynamic tool loading
- Agent routing system is extensible (just add new `elif` condition)

✅ **Integration Approach**:
- Add `agent_type="deep_research"` to database model
- Create `deep_research_agent.py` handler (mirror `web_research_agent.py`)
- Add routing in `agent_run_routes.py` (lines 578-614)
- Store configuration in JSON field (`deep_research_config`)
- Reuse existing tools (Tavily, MCP handler)

⚠️ **New Work Required**:
- Clone/adapt Open Deep Research source code
- Implement native search for Anthropic/OpenAI (if needed)
- Create UI for deep research agent configuration
- Add state types for deep research
- Create prompts for deep research stages (or reuse web_search)

### Immediate Next Steps

1. **Clone Open Deep Research Repository**
   ```bash
   git clone https://github.com/langchain-ai/open_deep_research.git
   cd open_deep_research
   # Examine src/ folder structure
   ```

2. **Examine Source Code**
   - Review `src/` folder structure
   - Identify core LangGraph nodes
   - Understand state management
   - Check configuration structure

3. **Create Proof of Concept**
   - Add `deep_research_config` field to Agent model
   - Create basic `deep_research_agent.py` skeleton
   - Add routing condition in `agent_run_routes.py`
   - Test with minimal implementation

4. **Iterate**
   - Integrate Open Deep Research core logic
   - Connect to existing tools (Tavily, MCP)
   - Add UI for configuration
   - Test end-to-end

### Estimated Effort

- **Phase 1 (Foundation)**: 2-3 days
- **Phase 2 (Core Integration)**: 5-7 days
- **Phase 3 (UI & Config)**: 3-4 days
- **Phase 4 (Testing)**: 2-3 days

**Total**: ~2-3 weeks for full integration

### Risk Assessment

**Low Risk**:
- Database schema changes (additive, no breaking changes)
- Routing changes (isolated, easy to test)
- UI changes (new agent type, doesn't affect existing)

**Medium Risk**:
- Open Deep Research integration complexity
- State management between systems
- Model configuration (4 models per agent)

**Mitigation**:
- Start with minimal viable integration
- Test incrementally
- Keep existing agents working (no breaking changes)

