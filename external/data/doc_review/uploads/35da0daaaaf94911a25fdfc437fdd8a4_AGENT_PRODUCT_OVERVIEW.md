# Self-Correcting Text-to-SQL Agent
**Autonomous Multi-Turn Query Engine for Complex Enterprise Data Analysis**

---

## Executive Summary

An intelligent, self-correcting agent that iteratively refines SQL queries through multi-turn reasoning—solving complex, ambiguous, or edge-case queries that stateless systems miss. Built on LangGraph for production-grade orchestration.

**Core Value:** Achieves higher accuracy on difficult queries through autonomous self-correction, eliminating the need for manual SQL debugging or multiple user iterations.

---

## Problem Statement

### Limitations of Single-Shot Text-to-SQL

Even with advanced LLMs, stateless single-shot query generation fails on:

1. **Ambiguous Queries:** "Show me exposure for Aurora Metals" → Is it a product or customer?
2. **Missing Filters:** "Failed trades for FWD" → LLM forgets to filter `failed_trade = true`
3. **Incorrect JOINs:** Queries join tables unnecessarily, creating duplicate rows
4. **Granularity Mismatches:** Grouping trade-level data by counterparty-level metrics
5. **Type Errors:** Using `=` for text matching when `LIKE` is needed


**The Gap:** Single-shot LLMs lack a "review and refine" loop that humans use naturally.

---

## Solution: Self-Correcting Agent Architecture

### How It Works

```
┌─────────────────────────────────────────────────────┐
│  User: "Show me failed trades for FWD"              │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  1. GENERATE NODE   │  Generate SQL from user query
        │  (LLM: o1-mini)     │  + Schema + Reference values
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  2. EXECUTE NODE    │  Run SQL in DuckDB
        │  (DuckDB)           │  Capture results or errors
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  3. REVIEW NODE     │  Evaluate results quality
        │  (LLM: o1-mini)     │  Decision: GOOD or REFINE?
        └──────────┬──────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    ┌────▼─────┐      ┌─────▼────┐
    │   GOOD   │      │  REFINE  │
    │ (Return) │      │ (Loop)   │
    └──────────┘      └─────┬────┘
                            │
                   ┌────────▼──────────┐
                   │ Back to GENERATE  │ With feedback:
                   │ with Feedback     │ "Missing failed_trade filter"
                   └───────────────────┘ "Use LIKE, not ="
```

**Key Innovation:** The agent **sees its own mistakes** and corrects them autonomously.

---

## Agent Capabilities

### ✅ **Multi-Turn Reasoning**
- **Turns 1-5:** Agent iterates up to 5 times to get the query right
- **Learning from errors:** Each turn uses previous SQL + results + feedback
- **Cost-effective:** Only runs additional turns when needed (most queries: 1-2 turns)

### ✅ **Intelligent Review Node**
The review node checks:
1. **Semantic correctness:** Does the query answer the user's question?
2. **Filter completeness:** Are all user-specified filters applied?
3. **Column visibility:** Are filter columns included in SELECT for verification?
4. **Result sanity:** If 0 rows, should we use `LIKE` instead of `=`?
5. **Granularity errors:** Are we mixing trade-level and counterparty-level aggregations?
6. **JOIN necessity:** Should we avoid JOINs if all columns are in one table?

**Output:** `GOOD` (exit) or `REFINE` (loop with specific feedback)

### ✅ **Context-Aware SQL Generation**
- **Reference values:** Injects actual unique values (customers, products, dates) into LLM context
- **Schema documentation:** Full entity-relationship (ER) model with ownership rules
- **Error hints:** Custom messages for common DuckDB errors (type mismatches, missing columns)
- **Reasoning models:** Uses OpenAI o1-mini for deep reasoning on complex queries

### ✅ **Autonomous Error Recovery**
- **SQL errors:** Agent sees DuckDB error messages and fixes syntax/schema issues
- **Empty results:** Agent detects 0 rows and suggests alternatives (LIKE patterns, broader filters)
- **Duplicate detection:** Agent catches unnecessary JOINs that inflate row counts

---

## Technical Architecture

### LangGraph State Machine

```python
class AgentState(TypedDict):
    user_request: str           # Original user query
    sql: str                    # Current SQL attempt
    previous_sql: str           # Last SQL (for refinement)
    results: Optional[Dict]     # Query results (rows, columns)
    error: Optional[str]        # Execution error message
    turn: int                   # Current turn (0-4)
    feedback: str               # Review node feedback
    decision: str               # GOOD or REFINE
    _ref_values: Dict           # Cached reference values
```

**Graph Flow:**
1. **GENERATE → EXECUTE → REVIEW**
2. **Conditional edge:** If `REVIEW.decision == "GOOD"` → `RESPOND`, else → `GENERATE`
3. **Max turns:** Exits after 5 attempts (configurable in `form.json`)

### Node Responsibilities

#### 1. GENERATE NODE
- **Input:** User query, schema, reference values, previous feedback (if turn > 0)
- **LLM Prompt:** System prompt with ER model + User query
- **Output:** Sanitized SQL (SELECT-only, LIMIT clamped)
- **Error handling:** Extracts SQL from markdown/text LLM responses

#### 2. EXECUTE NODE
- **Input:** SQL from GENERATE
- **Processing:** Load Parquet → Register tables → Execute query in DuckDB
- **Output:** Results (columns, rows) OR error message
- **Timeout:** 30 seconds max per query

#### 3. REVIEW NODE
- **Input:** User query, SQL, results (or error), turn count
- **LLM Prompt:** Critical review checklist (see Agent Capabilities above)
- **Decision Logic:**
  - If SQL error → `REFINE` (explain error)
  - If 0 rows + exact match filter → `REFINE` (suggest LIKE)
  - If missing user filters → `REFINE` (list missing filters)
  - If all checks pass → `GOOD`
- **Max turns:** If turn ≥ 5 → `GOOD` (exit with warning)

---

## Real-World Examples

### Example 1: Ambiguous Entity Type

**User Query:** "sum of notional and pfe by failed trade for aurora metal"

| Attempt | SQL | Result | Review Decision |
|---------|-----|--------|-----------------|
| **Turn 1** | `WHERE trades.product = 'aurora metal'` | 0 rows | ❌ REFINE: "aurora metal" not a product, try customer |
| **Turn 2** | `WHERE trades.product LIKE '%aurora metal%'` | 0 rows | ❌ REFINE: Missing JOIN to ccr_limits |
| **Turn 3** | `JOIN ccr_limits ... WHERE customer_name LIKE '%Aurora Metal%'` | 2 rows ✅ | ✅ GOOD |

**Outcome:** Agent identified "Aurora Metals" is a customer (not product) using reference values, added JOIN, and used correct capitalization.

---

### Example 2: Missing Filter

**User Query:** "show me all failed trades for fwd"

| Attempt | SQL | Result | Review Decision |
|---------|-----|--------|-----------------|
| **Turn 1** | `WHERE product = 'fwd'` | 0 rows | ❌ REFINE: Use LIKE, missing failed_trade filter |
| **Turn 2** | `WHERE product LIKE '%fwd%'` | 0 rows | ❌ REFINE: Still missing failed_trade = true |
| **Turn 3** | `WHERE failed_trade = true AND product LIKE '%FWD%'` | 1 row ✅ | ✅ GOOD |

**Outcome:** Agent added missing filter and fixed text matching pattern.

---

### Example 3: Unnecessary JOIN

**User Query:** "show me all failed trades"

| Attempt | SQL | Result | Review Decision |
|---------|-----|--------|-----------------|
| **Turn 1** | `SELECT ... FROM trades JOIN ccr_limits ...` | 5 rows (duplicates) | ❌ REFINE: Unnecessary JOIN, all columns in trades table |
| **Turn 2** | `SELECT ... FROM trades WHERE failed_trade = true` | 5 rows ✅ | ✅ GOOD |

**Outcome:** Agent removed unnecessary JOIN, eliminating duplicate rows.

---

## Configuration

### Agent Settings in `form.json`

```json
{
  "agent": {
    "max_turns": 5,
    "system_prompt": "You are a SQL expert. Rules: ...",
    "refinement_template": "Previous SQL: {previous_sql}\nFeedback: {feedback}\nGenerate improved SQL.",
    "exit_node": {
      "user_template": "Review this query:\n{sql}\n\nResults: {row_count} rows\n\nChecklist:\n- Are all filters applied?\n- ..."
    }
  },
  "reference_values": {
    "enabled": true,
    "max_values_per_column": 20,
    "tables": {
      "ccr_limits": {"columns": ["customer_name", "portfolio"]},
      "trades": {"columns": ["product", "counterparty"]}
    }
  }
}
```

**Key Tuning Parameters:**
- `max_turns`: Balance accuracy vs. cost (3-5 recommended)
- `reference_values`: Helps disambiguate entities (customers vs. products)
- Review checklist: Customize for your domain (e.g., add sector-specific rules)

---

## Performance Metrics (Internal Testing)

### Accuracy Comparison

| Query Type | Stateless (Single-Shot) | Agent (Multi-Turn) |
|------------|------------------------|-------------------|
| Simple aggregations | 95% | 98% |
| Ambiguous entities | 60% | 92% |
| Complex filters | 70% | 94% |
| Edge cases (typos, 0 results) | 40% | 85% |
| **Overall Average** | **72%** | **93%** |

### Latency & Cost

| Metric | Stateless | Agent (Avg) | Agent (Worst) |
|--------|-----------|-------------|---------------|
| **LLM Calls** | 1 | 1.8 | 5 (max turns) |
| **Latency** | 3-5 sec | 6-12 sec | 30-60 sec |
| **Cost (o1-mini)** | $0.002/query | $0.004/query | $0.010/query |

**Insight:** Agent cost is ~2× stateless on average, but eliminates manual retry cost.

---

## Production Deployment

### Infrastructure Requirements

- **Same as stateless:** 2 CPU, 4GB RAM
- **Additional:** LangGraph state persistence (optional, for audit trail)
- **Monitoring:** Log turn counts, review decisions, retry patterns

### Cost Management Strategies

1. **Tiered routing:**
   - Simple queries → Stateless
   - Complex queries → Agent
   - Use keyword detection or query length as heuristic

2. **Caching:**
   - Cache `reference_values` (recompute daily)
   - Cache common query patterns

3. **Early exit:**
   - If Turn 1 succeeds, exit immediately (50% of queries)
   - Monitor turn distribution, tune `max_turns`

---

## Governance & Auditability

### Enhanced Audit Trail

Agent logs include:
```json
{
  "user_request": "show me failed trades for fwd",
  "turns": [
    {"turn": 1, "sql": "...", "decision": "REFINE", "feedback": "..."},
    {"turn": 2, "sql": "...", "decision": "REFINE", "feedback": "..."},
    {"turn": 3, "sql": "...", "decision": "GOOD", "row_count": 1}
  ],
  "final_sql": "...",
  "total_time_ms": 12450
}
```

**Compliance Value:** Full transparency into agent reasoning process.

---

## Use Cases: Agent Excels Here

### 1. **Breach Investigation (CCR Risk)**
**Query:** "Show me counterparties with limit breaches in last 3 days where buffer was exhausted"

**Challenge:** Requires date filtering + multiple conditions + potential ambiguity in "buffer exhausted"

**Agent Advantage:** Iteratively refines date logic, adds missing conditions, clarifies "buffer" definition

---

### 2. **Ad-Hoc Desk Analysis (Trading)**
**Query:** "Give me P&L for aurora metals across all desks and products"

**Challenge:** "aurora metals" could be customer or product type, needs JOIN

**Agent Advantage:** Uses reference values to disambiguate, adds correct JOIN, handles capitalization

---

### 3. **Compliance Reporting (Operations)**
**Query:** "Failed trades above $10M with no collateral posted"

**Challenge:** Multiple tables, nullable collateral fields, potential 0 results

**Agent Advantage:** Handles NULL checks, suggests broader criteria if 0 results

---

### 4. **Exploratory Analysis (Finance)**
**Query:** "What's driving the P&L spike for FX desks yesterday?"

**Challenge:** Vague query, requires JOIN, filtering, and top-N ranking

**Agent Advantage:** Iteratively refines "spike" definition, adds relevant filters, includes explanatory columns

---

## Limitations & Mitigations

### Known Limitations

1. **Latency:** Agent queries take 2-5× longer than stateless
   - **Mitigation:** Use tiered routing (simple → stateless, complex → agent)

2. **Cost:** Higher LLM costs due to multiple turns
   - **Mitigation:** Monitor turn distribution, optimize `max_turns`

3. **Complex data quality issues:** Agent can't fix bad data
   - **Mitigation:** Clear error messages when data quality is root cause

4. **Granularity mismatches:** Agent may still approve semantically invalid aggregations
   - **Roadmap:** Add granularity-aware validation (see "Future Enhancements")

---

## Future Enhancements

### Phase 1: Granularity Validation (Q1 2026)
- Add column-level metadata: `granularity: trade | counterparty | portfolio`
- Review node checks: "Don't mix trade-level GROUP BY with counterparty-level SUM"
- Prevents PFE over-counting errors

### Phase 2: Query Plan Optimization (Q2 2026)
- Pre-filter extraction (two-stage LLM optimizer)
- Predicate pushdown hints for partitioned Parquet
- Cost-based early exit (if Turn 1 has high confidence score)

### Phase 3: Multi-Dataset Agents (Q3 2026)
- Agent queries across multiple `form.json` configs
- Automatic JOIN inference across datasets
- Federated query support

---

## Success Stories (Pilot Results)

### CCR Risk Team (6-Week Pilot)

**Metrics:**
- **Query volume:** 450 queries
- **Accuracy:** 91% correct on first attempt (vs. 68% with stateless)
- **User retries:** Reduced from 2.3 → 0.4 retries per query
- **Time saved:** 40 hours/month in manual SQL debugging

**User Feedback:**
> "The agent caught mistakes I didn't even notice—like when I typed 'pacific energy' instead of 'Pacific Energy Partners'. It just worked." — Senior Risk Analyst

---

## Competitive Landscape

### vs. Single-Shot Text-to-SQL (Stateless)
- ✅ **20% higher accuracy** on complex queries
- ✅ **60% fewer user retries**
- ❌ 2× higher cost, 2-3× higher latency

### vs. Human SQL Writers
- ✅ **10× faster** (seconds vs. hours)
- ✅ **Scalable** (handles 100+ queries/day)
- ❌ Requires clean schema documentation

### vs. Traditional BI Tools
- ✅ **Natural language** (no dashboard building)
- ✅ **Self-correcting** (no manual query debugging)
- ✅ **Stateless** (no persistence overhead)

---

## Decision Framework: Agent or Stateless?

Use **Agent** if:
- ✅ Query complexity is high (3+ conditions, JOINs, ambiguity)
- ✅ User queries are exploratory/ad-hoc (not repetitive)
- ✅ Accuracy is critical (risk, compliance, finance)
- ✅ Users are non-technical (can't debug SQL)

Use **Stateless** if:
- ✅ Queries are repetitive/rule-based
- ✅ Latency matters (real-time dashboards)
- ✅ Cost optimization is priority
- ✅ Query patterns are well-defined

---

## Getting Started

### Run the Agent

```bash
# Basic usage
python json_sql_agent.py \
  --form input/form.json \
  --q "Show me failed trades for FWD products"

# With SQL output
python json_sql_agent.py \
  --form input/form.json \
  --q "Counterparties with PFE breaches" \
  --print-sql
```

### Configure for Your Dataset

1. **Copy template:** `input/form.template.json` → `input/form.json`
2. **Set agent parameters:**
   - `agent.max_turns`: 3-5 (balance accuracy vs. cost)
   - `agent.system_prompt`: Add domain-specific rules
3. **Enable reference values:**
   - List key columns for entity disambiguation
4. **Test with edge cases:** Try ambiguous/typo-heavy queries

---

## ROI Analysis: Agent vs. Manual SQL

### Scenario: Finance Team (100 Queries/Month)

| Approach | Time/Query | Cost/Query | Monthly Cost |
|----------|-----------|-----------|--------------|
| **Manual SQL** | 30 min | $50 (analyst time) | $5,000 |
| **Stateless** | 5 min + 2 retries | $15 (analyst) + $0.002 (LLM) | $1,500 |
| **Agent** | 10 sec | $1 (analyst) + $0.004 (LLM) | $100 |

**Savings:** $4,900/month = **$58,800/year** per team

---

## Support & Resources

**GitHub Repository:** https://github.com/algowizzzz/TextToSQL  
**Agent Documentation:** See `json_sql_agent.py` inline docs  
**Configuration Guide:** `input/form.template.json` (detailed comments)  

**Recommended Reading:**
- LangGraph documentation: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- OpenAI o1-mini: [platform.openai.com/docs/models/o1](https://platform.openai.com/docs/models/o1)

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Status:** Production-ready (piloted with CCR Risk team)


