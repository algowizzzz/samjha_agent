# Agent UI Attributes Comparison

## Structured Agent (SQL/Parquet)

### Events Emitted (SSE)
1. **`run_started`** - Query initiated
   ```json
   {"event_type": "run_started", "payload": {"query": "user query"}}
   ```

2. **`decider_done`** - Planning complete
   ```json
   {"event_type": "decider_done", "payload": {}}
   ```

3. **`sql_generated`** - SQL query created
   ```json
   {"event_type": "sql_generated", "payload": {"sql": "SELECT ..."}}
   ```

4. **`results_ready`** - Query executed, results available
   ```json
   {"event_type": "results_ready", "payload": {"row_count": 42}}
   ```

5. **`final_response`** - LLM commentary ready
   ```json
   {"event_type": "final_response", "payload": {"response": "markdown text"}}
   ```

6. **`run_completed`** - Run finished
   ```json
   {"event_type": "run_completed", "payload": {"status": "success"}}
   ```

### Attributes Available in UI
- **`sql`** - Generated SQL query (from `sql_generated` event)
- **`row_count`** - Number of rows returned (from `results_ready` event)
- **`response`** - LLM-generated commentary/explanation (from `final_response` event)
- **`table_data`** - Structured table output (columns, rows) - stored in DB, loaded separately
- **`query_spec`** - Internal planning structure (not directly shown in UI)
- **`reasoning`** - Thinking traces (if `show_thinking=true`)

### UI Display
- Shows SQL query in collapsible section
- Displays table with columns/rows
- Shows LLM commentary as markdown
- Optional: Shows reasoning traces if enabled

---

## Web Research Agent (External)

### Events Emitted (SSE) - CURRENT
1. **`run_started`** - Research initiated
   ```json
   {"event_type": "run_started", "payload": {"query": "user query"}}
   ```

2. **`decider_done`** - Planning complete
   ```json
   {"event_type": "decider_done", "payload": {}}
   ```

3. **`sources_collected`** - Sources found
   ```json
   {"event_type": "sources_collected", "payload": {"count": 15}}
   ```

4. **`claims_extracted`** - Claims extracted
   ```json
   {"event_type": "claims_extracted", "payload": {"count": 53}}
   ```

5. **`conflicts_detected`** - Conflicts found (optional)
   ```json
   {"event_type": "conflicts_detected", "payload": {"count": 2}}
   ```

6. **`final_response`** - Final answer ready
   ```json
   {
     "event_type": "final_response",
     "payload": {
       "response": "markdown answer",
       "evidence_pack": {
         "sources": [...],
         "claims": [...],
         "conflicts": [...],
         "gaps": [...]
       }
     }
   }
   ```

7. **`run_completed`** - Run finished
   ```json
   {"event_type": "run_completed", "payload": {"status": "success"}}
   ```

### Attributes Available in UI - CURRENT
- **`response`** - Final synthesized answer (markdown)
- **`evidence_pack`** - Complete evidence structure
  - `sources` - Array of source objects (url, title, snippet, domain)
  - `claims` - Array of claim objects (text, confidence, source)
  - `conflicts` - Array of conflict objects (if any)
  - `gaps` - Array of gap objects (if any)
- **`count`** metrics - Sources count, claims count, conflicts count

### UI Display - CURRENT
- Shows final answer as markdown
- Displays evidence pack:
  - Sources list (with links)
  - Claims list (with confidence badges)
  - Conflicts list (if any)
  - Gaps list (if any)

---

## Missing Attributes for Web Research Agent

### What's Missing (compared to Structured Agent)

1. **`research_spec`** - Equivalent to `query_spec`
   - Should show: research question, search domains, depth settings
   - Currently: Not emitted to UI

2. **`search_queries`** - Equivalent to `sql_generated`
   - Should show: Actual search queries sent to Tavily
   - Currently: Not emitted to UI

3. **`reasoning`** - Thinking traces
   - Should show: Decider reasoning, Executor steps
   - Currently: Not emitted (even if `show_thinking=true`)

4. **`iteration_count`** - How many research iterations
   - Should show: "Iteration 1/3", "Iteration 2/3"
   - Currently: Not emitted to UI

5. **`research_plan`** - Planned search strategy
   - Should show: What searches will be performed
   - Currently: Not emitted to UI

---

## Recommended Additions

### New Events to Add

1. **`research_spec_generated`** - Research plan created
   ```json
   {
     "event_type": "research_spec_generated",
     "payload": {
       "research_spec": {
         "user_question": "...",
         "search_domains": [...],
         "research_depth": "standard"
       }
     }
   }
   ```

2. **`search_executed`** - Search query executed
   ```json
   {
     "event_type": "search_executed",
     "payload": {
       "query": "Apple Inc financial news",
       "results_count": 10,
       "iteration": 1
     }
   }
   ```

3. **`iteration_complete`** - Research iteration finished
   ```json
   {
     "event_type": "iteration_complete",
     "payload": {
       "iteration": 1,
       "total_iterations": 3,
       "sources_found": 10,
       "claims_extracted": 25
     }
   }
   ```

4. **`synthesis_started`** - Final answer generation started
   ```json
   {
     "event_type": "synthesis_started",
     "payload": {}
   }
   ```

### Enhanced Final Response
```json
{
  "event_type": "final_response",
  "payload": {
    "response": "markdown answer",
    "evidence_pack": {...},
    "research_spec": {...},
    "iteration_count": 2,
    "total_sources": 15,
    "total_claims": 53,
    "search_queries": [
      "Apple Inc financial news",
      "Apple Inc SEC filings 2024"
    ]
  }
}
```

---

## Summary

### Structured Agent Sends:
- ✅ SQL query
- ✅ Table output (columns, rows)
- ✅ Row count
- ✅ LLM commentary
- ✅ Reasoning (optional)

### Web Research Agent Currently Sends:
- ✅ Final answer (markdown)
- ✅ Evidence pack (sources, claims, conflicts, gaps)
- ✅ Count metrics
- ❌ Research spec (missing)
- ❌ Search queries (missing)
- ❌ Reasoning traces (missing)
- ❌ Iteration progress (missing)

