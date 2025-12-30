# Web Search Agent: Complete Flow & State Mapping
## Business PM Guide

---

## Executive Summary

This document maps **every component** from structured agents to web search agents, showing:
- What state/data flows through the system
- How Decider and Executor interact
- Where prompts are used
- How the "scratchpad" (state persistence) works
- Complete end-to-end flow

**Key Insight:** Web search agents use the **same architecture** as structured agents, but swap:
- `QuerySpec` → `ResearchSpec`
- SQL execution → Web search execution
- Data tables → Web sources

---

## 1. Component Mapping: Structured → Web Search

### A. State Types

| Structured Agent | Web Search Agent | What It Stores |
|------------------|------------------|----------------|
| `ControllerState` | `ResearchControllerState` | User query, conversation history, domain config, research spec, attempt count |
| `ExecutorState` | `ResearchExecutorState` | Research spec, evidence pack, sources seen, conflicts, executor report |
| `query_spec` | `research_spec` | Truth contract: what to research, scope, quality bar |
| `query_spec_status` | `research_spec_status` | Status tracker: what's known vs missing vs verified |
| `investigation_plan` | `research_plan` | Step-by-step plan: which tools to call, in what order |
| `domain_md` | `domain_content` | Domain config: allowed domains, research depth, source quality rules |

### B. Prompts

| Structured Agent | Web Search Agent | Purpose |
|------------------|------------------|---------|
| `decider.md` | `web_research_decider.md` | Gate: Creates ResearchSpec, decides ASK_USER/EXECUTE/BLOCK |
| `nl_to_sql_planner.md` | (Not needed - tools handle search) | N/A - Tavily tools execute directly |
| `query_result_evaluator.md` | `web_research_synthesis.md` | Final answer generation from evidence |
| `response_commentary.md` | (Same - reused) | Natural language response formatting |
| `ask_user_clarification.md` | (Same - reused) | User clarification questions |

### C. Tools

| Structured Agent | Web Search Agent | Purpose |
|------------------|------------------|---------|
| `list_dir` | (Not needed) | N/A |
| `inspect_table` | (Not needed) | N/A |
| `search_glossary` | (Not needed) | N/A |
| `nl_to_sql_planner` | (Not needed) | N/A |
| `execute_sql` | `tavily_web_search` | Execute search query |
| `tavily_research_search` | Execute deep research |
| `tavily_news_search` | Execute news search |
| `tavily_domain_search` | Execute domain-specific search |

---

## 2. Complete State Flow

### A. ResearchControllerState (Like ControllerState)

**What it is:** The "scratchpad" that persists across iterations

```python
ResearchControllerState = {
    # Inputs (from user/system)
    "user_query": "Verify that quantum error correction improved in 2024",
    "conversation_history": [
        {"query": "...", "response": "...", "status": "SUCCESS"}
    ],
    "domain_content": "# SEC Financial Research Agent\n## Allowed Domains...",  # From DB
    "agent_id": "sec_financial_agent",
    "agent_model": "claude-3-sonnet-20240229",
    
    # Research contract (created by Decider)
    "research_spec": {
        "user_question": "Verify that quantum error correction improved in 2024",
        "intent_type": "verify_claim",
        "scope": {
            "topic": "quantum error correction",
            "time_range": "2024",
            "entities": ["quantum error correction"]
        },
        "quality_bar": {
            "min_sources": 6,
            "source_types_required": ["academic", "news", "industry"]
        },
        "plan": [
            {"step": 1, "tool": "tavily_research_search", "args": {...}},
            {"step": 2, "tool": "tavily_domain_search", "args": {...}}
        ]
    },
    
    # Status tracker (what's known vs missing)
    "research_spec_status": {
        "user_question": {"status": "inferred", "source": "user", "blocks_execution": false},
        "scope.topic": {"status": "inferred", "source": "user", "blocks_execution": false},
        "scope.time_range": {"status": "inferred", "source": "user", "blocks_execution": false},
        "quality_bar.min_sources": {"status": "defaulted", "source": "domain_md", "blocks_execution": false}
    },
    
    # Iteration tracking
    "iteration_count": 0,
    "attempt_count": 0,
    "max_iterations": 2,  # From domain config
    
    # Evidence accumulation (grows across iterations)
    "evidence_pack": {
        "sources": [],
        "claims": [],
        "conflicts": [],
        "coverage_gaps": []
    },
    
    # Continuity (for follow-ups)
    "prior_research_spec": {},  # From last query
    "prior_research_spec_status": {},  # From last query
    
    # Executor feedback
    "last_executor_report": None,  # Filled after Executor runs
    
    # Policy limits
    "policy_limits": {
        "max_iterations": 2,
        "max_sources": 20,
        "max_api_calls": 10,
        "cost_limit_usd": 0.50
    },
    
    # UI options
    "show_thinking": false,
    "thinking_trace": None
}
```

### B. ResearchExecutorState (Like ExecutorState)

**What it is:** State during Executor execution (one iteration)

```python
ResearchExecutorState = {
    # Inputs (from Controller)
    "research_spec": {...},  # From ControllerState
    "research_spec_status": {...},  # From ControllerState
    "research_plan": [...],  # From research_spec.plan
    "evidence_pack": {...},  # Accumulated from prior iterations
    "sources_seen": ["url1", "url2"],  # Prevent duplicates
    "agent_data_folder": None,  # Not used for web search
    "policy_limits": {...},
    
    # Execution tracking
    "current_step": 1,
    "api_calls_made": 0,
    "estimated_cost_usd": 0.0,
    
    # Outputs (filled by Executor)
    "evidence_pack": {
        "sources": [
            {
                "id": "source_1",
                "title": "...",
                "url": "...",
                "domain": "arxiv.org",
                "score": 0.9,
                "source_type": "academic"
            }
        ],
        "claims": [
            {
                "claim_text": "Quantum error correction improved 50% in 2024",
                "supported_by": ["source_1", "source_2"],
                "confidence": 0.8
            }
        ],
        "conflicts": [
            {
                "claim1": "Improved 50%",
                "claim2": "Improved 20%",
                "severity": "high"
            }
        ],
        "coverage_gaps": ["Missing info on IBM approach"]
    },
    
    # Final report
    "executor_report": {
        "status": "SUCCESS",
        "evidence_pack": {...},
        "api_calls_made": 5,
        "estimated_cost_usd": 0.15,
        "last_error": None
    },
    
    # Early halt flag
    "halt_execution": false
}
```

---

## 3. Complete Flow: Step-by-Step

### Phase 1: User Query Arrives

```
User: "Verify that quantum error correction improved in 2024"
     ↓
[Controller receives query]
     ↓
Initialize ResearchControllerState:
- user_query = "Verify that quantum error correction improved in 2024"
- conversation_history = [] (first query)
- domain_content = Load from DB (agent_id)
- research_spec = {} (empty, will be filled by Decider)
- iteration_count = 0
- evidence_pack = {} (empty)
```

### Phase 2: Decider (Gate) - First Pass

```
[Controller calls Decider]
     ↓
Input to Decider:
- user_query
- conversation_history
- domain_content (SEC config, allowed domains, research depth)
- prior_research_spec = {} (empty, first query)
- last_executor_report = None
     ↓
[Decider prompt: web_research_decider.md]
     ↓
Decider reasoning:
1. Analyze query: "verify_claim" intent
2. Extract scope: topic="quantum error correction", time_range="2024"
3. Check domain config: min_sources=6, require authority sources
4. Create research_plan: 
   - Step 1: tavily_research_search (broad discovery)
   - Step 2: tavily_domain_search (authority verification)
5. Decide: EXECUTE (has enough info)
     ↓
Decider Output:
{
    "action": "EXECUTE",
    "query_type": "NEW_QUERY",
    "research_spec": {
        "user_question": "Verify that quantum error correction improved in 2024",
        "intent_type": "verify_claim",
        "scope": {
            "topic": "quantum error correction",
            "time_range": "2024",
            "entities": ["quantum error correction"]
        },
        "quality_bar": {
            "min_sources": 6,
            "source_types_required": ["academic", "news", "industry"]
        },
        "plan": [
            {"step": 1, "tool": "tavily_research_search", "args": {...}},
            {"step": 2, "tool": "tavily_domain_search", "args": {...}}
        ]
    },
    "research_spec_status": {...}
}
     ↓
[Controller updates state]
- research_spec = Decider output
- research_spec_status = Decider output
- iteration_count = 0
```

### Phase 3: Executor (Runner) - First Pass

```
[Controller calls Executor]
     ↓
Input to Executor:
- research_spec (from Decider)
- research_plan (from research_spec.plan)
- evidence_pack = {} (empty, first iteration)
- sources_seen = [] (empty)
     ↓
[Executor executes plan steps]
     ↓
Step 1: tavily_research_search
- Query: "quantum error correction 2024"
- Results: 10 sources
- Deduplicate: Remove duplicates
- Score sources: arxiv.org=0.9, nature.com=0.85, ...
- Extract claims: "Improved 50%", "Improved 20%", ...
     ↓
Step 2: tavily_domain_search
- Query: "quantum error correction improvements"
- Domains: ["arxiv.org", "nature.com", "ibm.com"]
- Results: 5 authority sources
- Merge with Step 1 results
     ↓
[Executor analyzes evidence]
- Cluster claims by theme
- Detect conflicts: "50%" vs "20%" (high severity)
- Identify gaps: Missing info on specific methods
     ↓
Executor Output (EvidencePack):
{
    "sources": [
        {"id": "s1", "title": "...", "url": "...", "score": 0.9},
        ...
    ],
    "claims": [
        {"claim_text": "Improved 50%", "supported_by": ["s1", "s2"]},
        {"claim_text": "Improved 20%", "supported_by": ["s3"]}
    ],
    "conflicts": [
        {
            "claim1": "Improved 50%",
            "claim2": "Improved 20%",
            "severity": "high"
        }
    ],
    "coverage_gaps": ["Missing info on IBM approach"]
}
     ↓
[Executor creates report]
executor_report = {
    "status": "SUCCESS",
    "evidence_pack": {...},
    "api_calls_made": 2,
    "estimated_cost_usd": 0.06
}
     ↓
[Controller updates state]
- evidence_pack = Executor output
- sources_seen = ["url1", "url2", ...] (tracked)
- last_executor_report = executor_report
- iteration_count = 1
```

### Phase 4: Decider (Gate) - Review & Iterate

```
[Controller calls Decider again - REVIEW PASS]
     ↓
Input to Decider:
- user_query (same)
- conversation_history (same)
- domain_content (same)
- research_spec (from prior Decider pass)
- evidence_pack (from Executor - NEW!)
- last_executor_report (from Executor - NEW!)
- iteration_count = 1
     ↓
[Decider analyzes evidence_pack]
     ↓
Decider reasoning:
1. Review evidence_pack:
   - 15 sources collected ✓
   - Claims extracted ✓
   - CONFLICT DETECTED: "50%" vs "20%" (high severity)
   - Gap: Missing IBM approach info
2. Check domain config:
   - High-severity conflicts require 3+ authority sources
   - Current: Only 2 sources for "50%" claim
3. Decision: EXECUTE (verify conflict)
     ↓
Decider Output:
{
    "action": "EXECUTE",
    "query_type": "RETRY",  # Iteration, not new query
    "research_spec": {
        "plan": [
            {
                "step": 1,
                "tool": "tavily_domain_search",
                "args": {
                    "query": "quantum error correction improvement percentage 2024",
                    "include_domains": ["arxiv.org", "nature.com", "ibm.com"]
                },
                "reason": "Verify conflicting claims with authority sources"
            }
        ]
    }
}
     ↓
[Controller updates state]
- research_spec.plan = New plan (conflict verification)
- iteration_count = 1 (increment after this)
```

### Phase 5: Executor (Runner) - Verification Pass

```
[Controller calls Executor again]
     ↓
Input to Executor:
- research_spec (updated plan from Decider)
- evidence_pack (from prior iteration - PRESERVED!)
- sources_seen (from prior iteration - PRESERVED!)
     ↓
[Executor executes new plan]
     ↓
Step 1: tavily_domain_search (authority verification)
- Query: "quantum error correction improvement percentage 2024"
- Domains: ["arxiv.org", "nature.com", "ibm.com"]
- Results: 5 new authority sources
- Check sources_seen: Skip duplicates
- Extract claims: "Improved 45%", "Improved 30%", ...
     ↓
[Executor merges with prior evidence]
- Add new sources to evidence_pack.sources
- Update claims: "Improved 50%" now has 4 sources (was 2)
- Update conflicts: Still conflicting, but more data
     ↓
Executor Output:
{
    "sources": [...],  # 20 total (15 + 5 new)
    "claims": [
        {"claim_text": "Improved 50%", "supported_by": ["s1", "s2", "s4", "s5"]},  # 4 sources
        {"claim_text": "Improved 20%", "supported_by": ["s3"]},  # 1 source
        {"claim_text": "Improved 45%", "supported_by": ["s6", "s7"]}  # 2 sources
    ],
    "conflicts": [
        {
            "claim1": "Improved 50%",
            "claim2": "Improved 20%",
            "severity": "medium"  # Downgraded (more sources for claim1)
        }
    ]
}
     ↓
[Controller updates state]
- evidence_pack = Merged evidence
- sources_seen = Updated (20 URLs)
- iteration_count = 2
- last_executor_report = New report
```

### Phase 6: Decider (Gate) - Final Decision

```
[Controller calls Decider - FINAL REVIEW]
     ↓
Input to Decider:
- evidence_pack (20 sources, claims, conflicts)
- iteration_count = 2
- max_iterations = 2 (from domain config)
     ↓
[Decider analyzes]
     ↓
Decider reasoning:
1. Review evidence:
   - 20 sources collected ✓ (exceeds min_sources=6)
   - Claims extracted ✓
   - Conflicts: Medium severity (acceptable)
   - Coverage: Good (no critical gaps)
2. Check iteration limit:
   - iteration_count = 2
   - max_iterations = 2
   - Reached limit
3. Decision: SUCCESS (synthesize answer)
     ↓
Decider Output:
{
    "action": "SUCCESS",  # Not EXECUTE - done researching
    "research_spec": {...},  # Final spec
    "evidence_pack": {...}  # Final evidence
}
```

### Phase 7: Synthesis (Final Answer)

```
[Controller calls Synthesis prompt]
     ↓
Input to Synthesis:
- user_query
- evidence_pack (final)
- research_spec
- output_format = "report"
     ↓
[Synthesis prompt: web_research_synthesis.md]
     ↓
Synthesis reasoning:
1. Review all claims
2. Identify consensus: "Improved 45-50%" (majority support)
3. Note disagreement: "Some sources say 20%" (minority)
4. Generate answer with citations
     ↓
Final Answer:
"""
# Research Report: Quantum Error Correction Improvements in 2024

## Summary
Based on 20 sources, quantum error correction showed significant improvements in 2024, with most sources indicating improvements of 45-50%.

## Key Findings
- **Primary consensus**: 45-50% improvement (supported by 6 sources)
- **Alternative view**: 20% improvement (supported by 1 source)
- **Areas of disagreement**: Exact percentage varies by method/implementation

## Sources
1. [Title](URL) - arxiv.org (academic)
2. [Title](URL) - nature.com (academic)
...
"""
     ↓
[Controller returns to user]
{
    "status": "SUCCESS",
    "finished_output": "...",
    "evidence_pack": {...},
    "sources": [...],
    "citations": [...]
}
```

---

## 4. State Persistence (Scratchpad)

### What Gets Saved Between Queries

**In Database (per conversation):**

```python
Conversation = {
    "id": "conv_123",
    "user_id": "user_456",
    "agent_id": "sec_financial_agent",
    "created_at": "...",
    "updated_at": "..."
}

# Per query (Run table)
Run = {
    "id": "run_789",
    "conversation_id": "conv_123",
    "agent_id": "sec_financial_agent",
    "user_query": "Verify that quantum error correction improved in 2024",
    "status": "success",
    "research_spec": {...},  # Saved for continuity
    "evidence_pack": {...},  # Saved for reference
    "created_at": "...",
    "finished_at": "..."
}
```

**In Memory (per session):**

```python
# Session storage (in-memory, cleared on logout)
session_state = {
    "conversation_id": "conv_123",
    "prior_research_spec": {...},  # From last query
    "prior_research_spec_status": {...},  # From last query
    "conversation_history": [
        {"query": "...", "response": "...", "status": "SUCCESS"}
    ]
}
```

### How Follow-Up Queries Work

```
User: "Tell me more about the IBM approach"
     ↓
[Controller loads prior state]
- prior_research_spec = From last query
- conversation_history = From last query
- evidence_pack = From last query (if relevant)
     ↓
[Decider detects FOLLOW_UP]
- query_type = "FOLLOW_UP"
- Merge with prior_research_spec
- Add new queries to plan
- Preserve existing evidence_pack
     ↓
[Executor executes]
- Uses existing evidence_pack as context
- Adds new sources
- Merges claims
```

---

## 5. Prompt Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
│  "Verify that quantum error correction improved in 2024"   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              CONTROLLER (Orchestrator)                      │
│  - Loads domain_content from DB                              │
│  - Initializes ResearchControllerState                       │
│  - Manages iteration loop                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DECIDER (Gate) - Pass 1                        │
│  Prompt: web_research_decider.md                            │
│  Input: user_query, domain_content, prior_spec              │
│  Output: ResearchSpec + Plan                                │
│  Decision: EXECUTE                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              EXECUTOR (Runner) - Pass 1                     │
│  Tools: tavily_research_search, tavily_domain_search        │
│  Input: research_plan                                       │
│  Output: EvidencePack (sources, claims, conflicts)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DECIDER (Gate) - Pass 2 (Review)               │
│  Prompt: web_research_decider.md                            │
│  Input: evidence_pack, conflicts, gaps                      │
│  Output: New plan (verify conflicts)                        │
│  Decision: EXECUTE (iterate)                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              EXECUTOR (Runner) - Pass 2                     │
│  Tools: tavily_domain_search (authority verification)         │
│  Input: Updated plan                                        │
│  Output: Updated EvidencePack (merged)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DECIDER (Gate) - Pass 3 (Final)                │
│  Prompt: web_research_decider.md                            │
│  Input: Final evidence_pack, iteration_count                │
│  Output: SUCCESS (synthesize)                                │
│  Decision: SUCCESS                                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SYNTHESIS (Final Answer)                       │
│  Prompt: web_research_synthesis.md                          │
│  Input: evidence_pack, research_spec                         │
│  Output: Final answer with citations                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    USER RECEIVES ANSWER                     │
│  - Report with findings                                     │
│  - Sources and citations                                    │
│  - Conflicts noted                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Key Differences: Structured vs Web Search

### A. State Differences

| Aspect | Structured Agent | Web Search Agent |
|--------|------------------|------------------|
| **Spec** | QuerySpec (SQL contract) | ResearchSpec (search contract) |
| **Plan** | Investigation plan (list_dir, inspect_table) | Research plan (tavily tools) |
| **Execution** | SQL query → Results table | Web search → EvidencePack |
| **Accumulation** | None (single query) | EvidencePack (accumulates across iterations) |
| **Status Tracker** | query_spec_status | research_spec_status |
| **Domain Config** | domain_md (tables, metrics) | domain_content (domains, depth) |

### B. Iteration Differences

**Structured Agent:**
- Usually 1 iteration (Decider → Executor → Done)
- Retries only on SQL errors
- No evidence accumulation

**Web Search Agent:**
- Usually 2-4 iterations (Discovery → Verification → Conflict Resolution)
- Iterates to resolve conflicts and fill gaps
- Evidence accumulates across iterations

### C. Output Differences

**Structured Agent:**
- SQL query
- Results table
- Natural language summary

**Web Search Agent:**
- EvidencePack (sources, claims, conflicts)
- Research report with citations
- Confidence levels per claim

---

## 7. Business PM Summary

### What Happens When User Asks a Question

1. **Controller receives query** → Loads agent config from DB
2. **Decider analyzes** → Creates ResearchSpec (what to research, how deep)
3. **Executor searches web** → Collects sources, extracts claims
4. **Decider reviews evidence** → Detects conflicts, gaps
5. **Executor verifies** → Resolves conflicts, fills gaps (if needed)
6. **Decider finalizes** → Decides when research is complete
7. **Synthesis generates answer** → Final report with citations

### What Gets Saved (Scratchpad)

- **Per query:** ResearchSpec, EvidencePack, sources, claims
- **Per conversation:** Prior specs, conversation history
- **Per agent:** Domain config, API keys, cost limits

### What You Control (Admin Panel)

- **Domain config file:** Where to search, how deep, what quality
- **Agent settings:** API keys, cost limits, default depth
- **Search scope:** Allowed/blocked domains (SEC-only, academic-only, etc.)

### Key Metrics to Track

- **Iterations per query:** Average 2-3 for standard research
- **Sources per query:** 6-20 for standard, 10-50 for deep
- **API calls per query:** 5-10 for standard, 10-20 for deep
- **Cost per query:** $0.15-0.30 for standard, $0.30-0.50 for deep
- **Time per query:** 1-2 minutes for standard, 3-5 minutes for deep

---

## 8. Next Steps

1. **Implement state types** (ResearchControllerState, ResearchExecutorState)
2. **Create prompts** (web_research_decider.md, web_research_synthesis.md)
3. **Build Executor nodes** (research discovery, verification, conflict resolution)
4. **Add state persistence** (save ResearchSpec, EvidencePack to DB)
5. **Test iteration loop** (Decider → Executor → Decider → ...)

---

## Conclusion

**Web search agents = Structured agents with:**
- ResearchSpec instead of QuerySpec
- EvidencePack instead of SQL results
- Multi-pass iteration instead of single-pass
- Conflict resolution instead of error retry

**Same architecture, different execution model.**

