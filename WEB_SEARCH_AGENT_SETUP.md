# Web Search Agent Setup & Testing Guide

## Implementation Summary

The web search agent feature has been fully implemented with the following components:

### 1. Database Migration
- **File**: `alembic/versions/0002_add_web_search_agent_fields.py`
- **Status**: Created (needs to be run when server has proper environment)
- **Columns Added**:
  - `tavily_api_key` (TEXT)
  - `search_scope_allowed_domains` (JSON)
  - `search_scope_blocked_domains` (JSON)
  - `default_research_depth` (VARCHAR(32))

### 2. System Prompts (6 prompts created)
All prompts are in `external/config/prompts/`:
- `web_research_decider.md` - Decision-making for research planning
- `web_research_synthesis.md` - Final answer synthesis
- `web_research_claim_extraction.md` - Extract claims from sources
- `web_research_conflict_detection.md` - Detect conflicts between sources
- `web_research_ask_user_clarification.md` - User clarification requests
- `web_research_response_commentary.md` - Response commentary

**Note**: These prompts are automatically imported on server startup via `web/app.py` line 88.

### 3. Core Agent Components
- **State Types**: `ResearchControllerState` and `ResearchExecutorState` in `external/agent/state_types.py`
- **Main Handler**: `external/agent/web_research_agent.py` - Orchestrates Decider/Executor loop
- **Decider**: `external/agent/web_research_decider.py` - Generates ResearchSpec
- **Executor Graph**: `external/agent/web_research_executor_graph.py` - LangGraph for execution
- **Executor Nodes**: `external/agent/web_research_executor_nodes.py` - Individual execution steps
- **Synthesis**: `external/agent/web_research_synthesis.py` - Final answer generation

### 4. Admin Panel Integration
- **File**: `web/templates/admin.html`
- **Features**:
  - Web-based Prompts section (editable)
  - External Agents section (create/manage)
  - Agent creation form with web search fields:
    - Tavily API Key
    - Allowed Domains
    - Blocked Domains
    - Default Research Depth (quick/standard/deep)

### 5. Chat Integration
- **File**: `web/templates/agents.html`
- Shows external agents under "External Data Agents (Web-based)"
- **File**: `external/routes/agent_run_routes.py`
- Routes external agents to `handle_web_research_query`

## Manual Testing Steps

### Step 1: Verify Prompts Are Imported
1. Start the server: `python3 run_server.py`
2. Navigate to: `http://localhost:8000/admin`
3. Login with: `admin` / `admin123`
4. Go to "System Prompts" → "Web-based Prompts"
5. Verify all 6 prompts are listed and editable

### Step 2: Create Financial News Agent
1. In Admin Panel, go to "Agent Instances" → "Web-based Agents"
2. Click "Create New Agent"
3. Fill in:
   - **Agent Type**: External Data/Web
   - **Name**: Financial News Research Agent
   - **Description**: Web research agent focused on financial news, market trends, and economic indicators
   - **LLM Model**: Claude 3 Sonnet
   - **Domain File**: Upload or create `financial_news_domain.md` with:
     ```
     # Financial News Research Agent Domain Configuration
     
     ## Authority Domains
     - sec.gov
     - federalreserve.gov
     - treasury.gov
     - bloomberg.com
     - reuters.com
     - wsj.com
     - ft.com
     
     ## Search Scope
     - Allowed: sec.gov, federalreserve.gov, treasury.gov, bloomberg.com, reuters.com, wsj.com, ft.com
     - Blocked: reddit.com, twitter.com, facebook.com
     
     ## Research Depth
     - Default: Standard (2 iterations, 6-20 sources)
     ```
   - **Tavily API Key**: (optional, can be set later)
   - **Allowed Domains**: `sec.gov,federalreserve.gov,treasury.gov,bloomberg.com,reuters.com,wsj.com,ft.com`
   - **Blocked Domains**: `reddit.com,twitter.com,facebook.com`
   - **Default Research Depth**: Standard
4. Click "Create Agent"

### Step 3: Test Agent in Chat
1. Navigate to: `http://localhost:8000/agents`
2. Find "External Data Agents (Web-based)" section
3. Select "Financial News Research Agent"
4. Enter a test query: "What are the latest trends in financial markets in 2024?"
5. Click "Send"
6. Observe:
   - SSE events streaming
   - Research plan generation
   - Source collection
   - Final synthesized answer

### Step 4: Verify API Endpoints
Using curl or Postman (with session cookie from browser):

```bash
# List agents
curl -b cookies.txt "http://localhost:8000/api/admin/agents"

# Get specific agent
curl -b cookies.txt "http://localhost:8000/api/admin/agents/financial-news-research-agent"

# Create a run
curl -b cookies.txt -X POST "http://localhost:8000/api/agents/financial-news-research-agent/runs" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest trends in financial markets?"}'

# Get run events
curl -b cookies.txt "http://localhost:8000/api/runs/{run_id}/events"
```

## Expected Behavior

1. **Decider Phase**: Generates ResearchSpec with:
   - Research question
   - Research plan (list of Tavily tool calls)
   - Expected sources count
   - Research depth

2. **Executor Phase**: Executes research plan:
   - Calls Tavily tools (web_search, news_search, research_search, domain_search)
   - Collects sources
   - Extracts claims from sources
   - Detects conflicts
   - Scores source quality

3. **Synthesis Phase**: Generates final answer:
   - Synthesizes evidence
   - Cites sources
   - Handles conflicts
   - Provides confidence level

4. **Iterative Loop**: If gaps/conflicts detected:
   - Decider plans follow-up research
   - Executor collects additional sources
   - Process repeats until SUCCESS or max_iterations

## Troubleshooting

### Prompts Not Imported
- Check server logs for import errors
- Manually trigger: Server restarts automatically import prompts
- Verify files exist in `external/config/prompts/web_research_*.md`

### Agent Creation Fails
- Check database migration ran successfully
- Verify Tavily API key is valid (if provided)
- Check server logs for errors

### Agent Run Fails
- Verify Tavily tools are registered: Check `external/config/tools/tavily_*.json`
- Check Tavily API key is set (agent-specific or environment variable)
- Verify domain file is uploaded
- Check server logs for detailed errors

### No Sources Found
- Verify allowed domains are correct
- Check Tavily API key is valid
- Verify search query is clear and specific

## Next Steps

1. **Run Migration**: When server environment is available:
   ```bash
   alembic upgrade head
   ```

2. **Test with Real Tavily API Key**: 
   - Get API key from https://tavily.com
   - Set in agent configuration
   - Test with real queries

3. **Monitor Performance**:
   - Check iteration counts
   - Monitor source quality
   - Review conflict detection accuracy

4. **Customize Prompts**:
   - Edit prompts via admin panel
   - Test different research depths
   - Adjust source quality thresholds

## Files Created/Modified

### New Files
- `external/config/prompts/web_research_*.md` (6 files)
- `external/agent/web_research_agent.py`
- `external/agent/web_research_decider.py`
- `external/agent/web_research_executor_graph.py`
- `external/agent/web_research_executor_nodes.py`
- `external/agent/web_research_synthesis.py`
- `alembic/versions/0002_add_web_search_agent_fields.py`

### Modified Files
- `external/agent/state_types.py` - Added ResearchControllerState, ResearchExecutorState
- `external/agent/persistence.py` - Added web search prompt metadata
- `core/db/models.py` - Added web search fields to Agent model
- `external/routes/agent_run_routes.py` - Added routing for external agents
- `web/templates/admin.html` - Added web search UI
- `web/templates/agents.html` - Added external agents listing
- `routes/admin_routes.py` - Added external agent creation support

## Summary

The web search agent is fully implemented and ready for testing. All components are in place:
- ✅ Database schema (migration ready)
- ✅ System prompts (6 prompts)
- ✅ Core agent logic (Decider/Executor/Synthesis)
- ✅ Admin panel integration
- ✅ Chat interface integration
- ✅ API endpoints

To test, start the server and use the admin panel to create an agent, then test it in the chat interface.


