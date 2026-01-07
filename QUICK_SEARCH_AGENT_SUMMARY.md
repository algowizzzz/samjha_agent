# Quick Search Agent - Financial News Implementation

## Overview
A new **Quick Search Agent** type that performs direct Tavily searches focused on financial news entities (sectors, counterparties, regulatory bodies, etc.).

## Key Features
- **Fast, Direct Search**: No iterations, no planning - just query → search → format → return
- **Financial News Focus**: Pre-configured with 30+ financial news domains
- **Entity-Oriented**: Optimized for searching entities like sectors, counterparties, regulatory bodies
- **Domain Restrictions**: Only searches financial news sources, blocks social media

## Financial News Domains Included

### Major Financial News Outlets
- Bloomberg, Reuters, WSJ, Financial Times, CNBC, MarketWatch
- Yahoo Finance, Investing.com, Seeking Alpha, Barron's, Forbes, Business Insider

### Regulatory Bodies & Official Sources
- SEC, Federal Reserve, FDIC, OCC (US)
- OSFI (Canada), Bank of England, ECB, BIS, FSB

### Financial Data & Analytics
- Morningstar, Fitch, Moody's, S&P Global

### Industry Publications
- American Banker, Banking Journal, Risk.net, The Banker

### Regional Financial News
- SCMP (Asia), AFR (Australia)

## Implementation Details

### Files Created/Modified

1. **`external/agent/quick_search_agent.py`** (NEW)
   - Handler function: `handle_quick_search_query()`
   - Domain lists: `FINANCIAL_NEWS_DOMAINS`, `BLOCKED_DOMAINS`
   - Configuration loader: `load_quick_search_config()`
   - Response formatter: `format_quick_search_response()`

2. **`external/core/db/models.py`** (MODIFIED)
   - Added `quick_search_config: Mapped[Optional[dict]]` column to `Agent` model

3. **`external/agent/persistence.py`** (MODIFIED)
   - Updated `get_agent_db()` to return `quick_search_config`
   - Updated `create_agent_db()` to accept `quick_search_config` parameter
   - Added support for `quick_search_config` in agent creation

4. **`external/routes/agent_run_routes.py`** (MODIFIED)
   - Added routing for `agent_type == "quick_search"`
   - Added result handling for quick search responses
   - Emits events: `sources_collected`, `final_response`, `run_completed`

5. **`external/web/templates/agents.html`** (MODIFIED)
   - Added "Quick Search Agents" section in UI
   - Added JavaScript filtering and rendering for `quick_search` agent type

6. **`add_quick_search_column_pg.py`** (NEW)
   - Script to add `quick_search_config` JSONB column to PostgreSQL

## Configuration

### Default Quick Search Config
```json
{
  "max_results": 5,
  "search_depth": "basic",
  "include_answer": true,
  "include_domains": [...30+ financial news domains...],
  "exclude_domains": ["reddit.com", "twitter.com", "x.com", ...]
}
```

### Agent-Specific Override
Agents can override default config via `quick_search_config` JSON field in DB:
- `max_results`: Number of results (default: 5)
- `search_depth`: "basic" or "advanced" (default: "basic")
- `include_answer`: Include AI-generated summary (default: true)
- `include_domains`: Override allowed domains list
- `exclude_domains`: Override blocked domains list

## Usage

### Creating a Quick Search Agent
1. Create agent with `agent_type="quick_search"`
2. Set `tavily_api_key` (required)
3. Optionally set `quick_search_config` for custom settings

### Example Query
- "Latest news on Apple stock"
- "Regulatory updates from SEC"
- "Banking sector analysis"
- "Counterparty risk for JPMorgan"

## Response Format

1. **AI-Generated Answer** (if available from Tavily)
2. **Key Information** section (entity-focused)
3. **Sources** list with:
   - Numbered citations with links
   - Relevance scores
   - Brief excerpts
4. **Summary footer** with result count

## Differences from Web Research Agent

| Feature | Quick Search | Web Research |
|---------|-------------|--------------|
| Architecture | Direct search | Controller/Decider/Executor |
| Iterations | None (single pass) | 2-4 iterations |
| Planning | None | ResearchSpec planning |
| Evidence Quality | Basic | Strict quality checks |
| Domain MD | Not used | Required |
| Speed | Very fast | Slower (comprehensive) |
| Use Case | Quick entity lookups | Deep research |

## Next Steps

1. Run `add_quick_search_column_pg.py` to add column to PostgreSQL
2. Create a test quick search agent via Admin Panel
3. Configure Tavily API key for the agent
4. Test with entity-focused queries

