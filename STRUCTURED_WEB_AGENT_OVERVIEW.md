# Structured Web Agent - Architecture & Implementation Status

## Overview

The system implements **two agent types** using the same **Decider/Executor architecture**:

1. **Structured Data Agent** (`parquet_agent.py`) - Queries structured data (Parquet/CSV files) via SQL
2. **Web Research Agent** (`web_research_agent.py`) - Performs web research using Tavily API

Both follow the same architectural pattern but use different tools and data contracts.

---

## Architecture: Decider/Executor Pattern

### High-Level Flow

```
User Query
    ↓
Controller Loop
    ↓
┌─────────────────┐
│   DECIDER       │  ← LLM-only reasoning
│   (Gate)        │     - Analyzes query
│                 │     - Fills Query Spec
│                 │     - Decides: ASK_USER | EXECUTE | BLOCK
└─────────────────┘
    ↓
Action Router
    ├─ ASK_USER → Return question to UI
    ├─ BLOCK → Return error
    └─ EXECUTE → Executor Graph
                    ↓
            ┌─────────────────┐
            │   EXECUTOR      │  ← Tool-driven execution
            │   (Worker)      │     - Runs investigation tools
            │                 │     - Generates SQL (structured) or searches web
            │                 │     - Validates & executes
            │                 │     - Evaluates results
            └─────────────────┘
                    ↓
            SUCCESS or ERROR
                    ↓
            (If ERROR) → Loop back to Decider with error report
```

### Key Principles

1. **Separation of Concerns**
   - **Decider**: LLM-only reasoning, no tool calls, produces execution contract
   - **Executor**: Deterministic tool execution, follows Decider's plan

2. **State Management**
   - `ControllerState`: Orchestration state (query, history, spec, status)
   - `ExecutorState`: Execution state (plan, SQL, results, errors)

3. **Retry Loop**
   - Controller manages retries (max_attempts)
   - On ERROR, Decider receives `last_executor_report` and replans
   - Preserves verified context across retries

---

## Structured Data Agent (Parquet Agent)

### Components

#### 1. Controller (`parquet_agent.py`)
- **Entry Point**: `handle_query()`
- **Responsibilities**:
  - Initialize state from user query + conversation history
  - Loop: Decider → Executor → (retry if ERROR)
  - Enforce `max_attempts` policy
  - Render responses (ASK_USER, SUCCESS, BLOCK, ERROR)

#### 2. Decider (`decider.py`)
- **Function**: `run_decider(state: ControllerState) -> dict`
- **Output Schema**: `decider_output.json` (validated)
- **Key Outputs**:
  - `action`: "ASK_USER" | "EXECUTE" | "BLOCK"
  - `query_spec`: Structured contract (business_question, dimensions, metrics, filters, etc.)
  - `query_spec_status`: Gap tracker (missing/verified/inferred/defaulted)
  - `investigation_plan`: List of tools to run (gap-driven)

#### 3. Executor Graph (`executor_graph.py`)
- **LangGraph Implementation**: 6 linear nodes
- **Nodes**:
  1. `investigation_node`: Runs tools from investigation_plan
  2. `sql_generation_node`: Generates SQL from complete spec
  3. `safety_validation_node`: Validates SQL (policy, limits, forbidden patterns)
  4. `execution_node`: Executes SQL via DuckDB
  5. `evaluation_node`: Evaluates results (sanity checks)
  6. `outcome_node`: Builds executor_report

#### 4. Tools (via `ToolsRegistry`)
- `list_dir`: List files in data folder
- `inspect_table`: Get table schema
- `preview_rows`: Preview sample data
- `search_glossary`: Resolve metric/dimension names
- `nl_to_sql_planner`: Generate SQL from spec
- `sql_plan_updater`: Update SQL plan
- `query_safety_validator`: Validate SQL safety
- `execute_sql`: Execute SQL query
- `query_result_evaluator`: Evaluate results

### Data Contracts

#### Query Spec (Table 9)
```python
{
    "business_question": str,
    "dimensions": List[str],
    "metrics": List[str],
    "filters": List[Dict],
    "start_table": {"path": str, "grain": str},
    "joins": List[Dict],
    "time": {"column": str, "range": Dict},
    "aggregation_plan": str,
    "validation_checks": List[str]
}
```

#### Query Spec Status (Table 10)
```python
{
    "dimensions": {"status": "verified|inferred|missing", "source": "domain_md|tool_result|user"},
    "metrics": {"status": "...", "source": "..."},
    # ... tracks every field in query_spec
}
```

---

## Web Research Agent

### Components

#### 1. Controller (`web_research_agent.py`)
- **Entry Point**: `handle_web_research_query()`
- **Similar structure** to Parquet Agent but:
  - Uses `ResearchControllerState`
  - Manages `iteration_count` (for deep research)
  - Accumulates `evidence_pack` across iterations

#### 2. Decider (`web_research_decider.py`)
- **Output Schema**: `research_decider_output.json`
- **Key Outputs**:
  - `action`: "ASK_USER" | "EXECUTE" | "BLOCK"
  - `research_spec`: Research contract (user_question, intent_type, scope, quality_bar)
  - `research_spec_status`: Gap tracker
  - `research_plan`: List of Tavily search tool calls

#### 3. Executor Graph (`web_research_executor_graph.py`)
- **Nodes**:
  1. `search_node`: Executes Tavily searches
  2. `evidence_extraction_node`: Extracts claims from sources
  3. `conflict_detection_node`: Detects conflicting claims
  4. `gap_analysis_node`: Identifies information gaps
  5. `synthesis_node`: Synthesizes final answer

#### 4. Tools
- Tavily search tools (registered via ToolsRegistry)
- Evidence extraction (LLM-based)
- Conflict detection (LLM-based)

### Data Contracts

#### Research Spec
```python
{
    "user_question": str,
    "intent_type": "factual|analytical|comparative",
    "scope": {"domains": List[str], "depth": str},
    "quality_bar": {"min_sources": int, "authority_required": bool},
    "constraints": Dict
}
```

#### Evidence Pack
```python
{
    "sources": List[Dict],  # URLs, titles, snippets
    "claims": List[Dict],   # Extracted claims with confidence
    "conflicts": List[Dict], # Conflicting claims
    "gaps": List[Dict]      # Information gaps
}
```

---

## Implementation Status

### ✅ Completed

1. **Core Architecture**
   - ✅ Decider/Executor separation
   - ✅ Controller loop with retry logic
   - ✅ State management (ControllerState, ExecutorState)
   - ✅ Schema validation (JSON schemas for Decider outputs)

2. **Structured Data Agent**
   - ✅ Decider implementation (`decider.py`)
   - ✅ Executor graph (LangGraph + fallback sequential)
   - ✅ All 9 tools implemented
   - ✅ SQL generation, validation, execution
   - ✅ Result evaluation
   - ✅ Follow-up query detection (FOLLOW_UP query type)
   - ✅ Continuity packet for cross-turn state

3. **Web Research Agent**
   - ✅ Decider implementation (`web_research_decider.py`)
   - ✅ Executor graph
   - ✅ Evidence accumulation across iterations
   - ✅ Conflict detection
   - ✅ Gap analysis
   - ✅ Final synthesis

4. **Infrastructure**
   - ✅ Tools registry system
   - ✅ Agent persistence (database)
   - ✅ Session management
   - ✅ API routes (`agent_run_routes.py`)
   - ✅ UI integration (admin panel, chat interface)

### 🚧 Partially Implemented

1. **Error Handling**
   - ✅ Basic error reporting
   - ⚠️ Some edge cases may need refinement
   - ⚠️ Error recovery could be more sophisticated

2. **Policy Enforcement**
   - ✅ Basic policy limits (max_attempts, max_rows)
   - ⚠️ Some advanced policies may need hardening

3. **Tool Capabilities**
   - ✅ All core tools implemented
   - ⚠️ Some tools may need refinement based on usage

### 📋 Known Limitations / TODOs

1. **LangGraph Dependency**
   - Fallback sequential executor exists, but LangGraph is preferred
   - Ensure LangGraph is properly installed in production

2. **Domain Configuration**
   - Domain files (`domain_md`) are critical for agent behavior
   - Need clear documentation on domain file format

3. **Testing**
   - Unit tests for individual components
   - Integration tests for full flows
   - End-to-end tests with real queries

4. **Performance**
   - LLM call optimization (caching, batching)
   - SQL query optimization
   - Tool execution parallelization where possible

5. **Observability**
   - Enhanced logging and tracing
   - Metrics collection
   - Debug mode for troubleshooting

---

## Key Files Reference

### Core Agent Files
- `external/agent/parquet_agent.py` - Structured data agent controller
- `external/agent/web_research_agent.py` - Web research agent controller
- `external/agent/decider.py` - Structured data decider
- `external/agent/web_research_decider.py` - Web research decider
- `external/agent/executor_graph.py` - Structured data executor graph
- `external/agent/web_research_executor_graph.py` - Web research executor graph
- `external/agent/executor_nodes.py` - Executor node implementations
- `external/agent/web_research_executor_nodes.py` - Web research executor nodes

### State & Schemas
- `external/agent/state_types.py` - TypedDict definitions
- `external/agent/schemas.py` - Legacy schemas (AgentState)
- `external/agent/schema_validators.py` - JSON schema validators

### Infrastructure
- `external/agent/persistence.py` - Database persistence
- `external/agent/session_manager.py` - Session management
- `external/routes/agent_run_routes.py` - API endpoints
- `tools/tools_registry.py` - Tool registry system

### Configuration
- `external/config/prompts/decider.md` - Decider prompt template
- `external/config/tools/*.json` - Tool configurations
- `external/config/domains/*_domain.md` - Domain configurations

---

## How to Use

### Structured Data Agent

```python
from external.agent.parquet_agent import handle_query

result = handle_query(
    user_query="Show me total sales by country",
    conversation_history=[],
    agent_id="agent_123",
    show_thinking=False
)

# Result status: "ASK_USER" | "SUCCESS" | "BLOCK" | "ERROR"
if result["status"] == "SUCCESS":
    print(result["finished_output"])
    print(result["final_sql"])
    print(result["results"])  # Table data
```

### Web Research Agent

```python
from external.agent.web_research_agent import handle_web_research_query

result = handle_web_research_query(
    user_query="What are the latest trends in AI?",
    conversation_history=[],
    agent_id="web_agent_123"
)

# Result status: "ASK_USER" | "SUCCESS" | "BLOCK" | "ERROR"
if result["status"] == "SUCCESS":
    print(result["final_answer"])
    print(result["evidence_pack"]["sources"])
```

---

## Architecture Benefits

1. **Deterministic Execution**: Executor follows Decider's plan, reducing hallucinations
2. **Clear Separation**: Reasoning (Decider) vs. Execution (Executor)
3. **Retry Logic**: Automatic retry with error context
4. **Extensibility**: Easy to add new tools or agents following the same pattern
5. **State Preservation**: Follow-up queries inherit verified context
6. **Schema Validation**: JSON schemas ensure contract compliance

---

## Next Steps / Recommendations

1. **Testing**: Add comprehensive test coverage
2. **Documentation**: Document domain file format and tool capabilities
3. **Monitoring**: Add metrics and observability
4. **Optimization**: Performance tuning for LLM calls and SQL execution
5. **Error Recovery**: Enhance error handling and recovery strategies
6. **User Experience**: Improve ASK_USER questions and error messages

