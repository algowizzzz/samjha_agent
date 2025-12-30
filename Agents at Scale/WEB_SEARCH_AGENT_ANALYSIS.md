# Web Search Agent Deep Research Analysis

## Executive Summary

Your plan is **architecturally sound** and correctly adapts the Decider+Executor pattern. However, there are **critical gaps** in state management, error handling, cost control, and reasoning architecture that will cause production failures if not addressed.

**Key Insight:** Deep research ≠ simple search. The difference is **iterative verification, conflict resolution, and evidence synthesis**—not just more API calls.

---

## 1. Plan Strengths ✅

### A. Correct Pattern Reuse
- ✅ Decider (Gate) + Executioner (Runner) split
- ✅ ResearchSpec as truth contract (analogous to QuerySpec)
- ✅ Two-pass research loop (Discovery → Verification)
- ✅ Evidence pack structure

### B. Good Architectural Decisions
- ✅ Capability mapping (4 Tavily tools as primitives)
- ✅ Source quality heuristics
- ✅ Deduplication strategy
- ✅ Citations contract

---

## 2. Critical Gaps (What You Missed) ❌

### A. State Management Across Iterations

**Problem:** Your plan doesn't specify how to track research state across multiple Decider→Executor loops.

**What's Missing:**
```python
# ResearchState (analogous to ControllerState)
{
    "user_query": str,
    "research_spec": ResearchSpec,
    "research_spec_status": ResearchSpecStatus,  # ← NEW
    "evidence_pack": EvidencePack,  # ← Accumulated across iterations
    "iteration_count": int,
    "sources_seen": Set[str],  # ← Prevent duplicate searches
    "claims_tracked": Dict[str, ClaimEvidence],  # ← Track claim→sources mapping
    "conflicts_pending": List[Conflict],  # ← Unresolved conflicts
    "open_questions": List[str],  # ← Questions that need more research
    "last_executor_report": ExecutorReport,
    "policy_limits": {
        "max_iterations": 3,
        "max_sources": 50,
        "max_api_calls": 20,
        "cost_limit_usd": 5.00
    }
}
```

**Why Critical:** Without state tracking, you'll:
- Re-search the same queries
- Lose context between iterations
- Fail to resolve conflicts systematically

---

### B. Error Handling & Resilience

**Problem:** Web APIs fail more often than SQL queries (rate limits, network errors, API key issues).

**What's Missing:**

1. **Retry Strategy with Exponential Backoff**
   ```python
   # In executor_nodes.py (web_search_research_node)
   MAX_RETRIES = 3
   BACKOFF_BASE = 2  # seconds
   
   for attempt in range(MAX_RETRIES):
       try:
           result = tavily_tool.execute(args)
           break
       except RateLimitError:
           wait_time = BACKOFF_BASE ** attempt
           time.sleep(wait_time)
       except APIKeyError:
           return {"status": "BLOCK", "reason": "API key invalid"}
       except NetworkError:
           if attempt < MAX_RETRIES - 1:
               continue
           return {"status": "ERROR", "reason": "Network failure"}
   ```

2. **Graceful Degradation**
   - If `tavily_research_search` fails, fall back to `tavily_web_search`
   - If all Tavily tools fail, return partial results from cache
   - If API key missing, use demo mode with clear warnings

3. **Cost Tracking**
   ```python
   # Track API costs per agent instance
   cost_tracker = {
       "api_calls": 0,
       "estimated_cost_usd": 0.0,
       "limit_exceeded": False
   }
   # Tavily pricing: ~$0.01 per search (estimate)
   # Enforce limits before making calls
   ```

---

### C. Caching Strategy

**Problem:** Users will ask similar questions repeatedly. Re-searching wastes API calls and time.

**What's Missing:**

1. **Query Cache**
   ```python
   # Cache key: normalized query + time_range + topic
   cache_key = f"{normalize_query(query)}_{time_range}_{topic}"
   
   # Check cache before API call
   cached_result = cache.get(cache_key)
   if cached_result and not force_refresh:
       return cached_result
   
   # Store in cache after API call (TTL: 24 hours for news, 7 days for general)
   cache.set(cache_key, result, ttl=ttl_by_topic)
   ```

2. **Source Deduplication Cache**
   - Track URLs already fetched
   - Skip re-fetching same source in same research session
   - Cross-session dedup (optional, privacy-sensitive)

---

### D. Follow-Up Query Handling

**Problem:** Your plan doesn't specify how to handle "tell me more about X" after initial research.

**What's Missing:**

1. **Follow-Up Detection (Similar to Structured Agent)**
   ```python
   # In decider.py (web_search_decider)
   follow_up_signals = [
       "tell me more about",
       "what about",
       "also search for",
       "find sources on",
       "verify that"
   ]
   
   # If follow-up detected:
   # - Merge with prior ResearchSpec
   # - Add new queries to plan
   # - Preserve existing evidence_pack
   ```

2. **Context Preservation**
   - Keep `evidence_pack` from prior query
   - Add new sources to existing claims
   - Merge conflicts rather than replacing

---

### E. Domain-Specific Knowledge

**Problem:** Medical, legal, and financial queries need special handling (authority requirements, recency rules).

**What's Missing:**

1. **Domain Rules in ResearchSpec**
   ```json
   {
     "domain": "medical|legal|financial|general",
     "authority_requirements": {
       "medical": ["pubmed.gov", "nih.gov", "who.int"],
       "legal": ["supremecourt.gov", "law.cornell.edu"],
       "financial": ["sec.gov", "federalreserve.gov"]
     },
     "recency_rules": {
       "medical": "last 5 years",
       "financial_regulations": "last 12 months"
     }
   }
   ```

2. **Domain-Aware Source Scoring**
   - Boost authority domains in scoring
   - Penalize non-authoritative sources for sensitive domains
   - Require minimum authority sources before finalizing

---

### F. Conflict Resolution Logic

**Problem:** Your plan mentions conflict resolution but doesn't specify the algorithm.

**What's Missing:**

1. **Conflict Detection**
   ```python
   # In executor_nodes.py
   def detect_conflicts(claims: List[Claim]) -> List[Conflict]:
       conflicts = []
       for claim1 in claims:
           for claim2 in claims:
               if claim1.claim_text != claim2.claim_text:
                   # Check if they contradict
                   if contradicts(claim1, claim2):
                       conflicts.append(Conflict(
                           claim1=claim1,
                           claim2=claim2,
                           sources1=claim1.supported_by,
                           sources2=claim2.supported_by,
                           severity="high|medium|low"
                       ))
       return conflicts
   ```

2. **Resolution Strategy**
   ```python
   # In decider.py (after executor report)
   if conflicts:
       # High severity: require 3+ independent sources
       if max(c.severity for c in conflicts) == "high":
           return {
               "action": "EXECUTE",
               "research_spec": {
                   "plan": [
                       {"tool": "tavily_domain_search", 
                        "args": {"include_domains": authority_domains}},
                       {"tool": "tavily_research_search",
                        "args": {"max_results": 10}}
                   ],
                   "min_sources": 3
               }
           }
   ```

---

### G. Reasoning Architecture (Critical for Deep Research)

**Problem:** Your plan mentions "reasoning is key" but doesn't specify how reasoning differs between simple search and deep research.

**What's Missing:**

#### Simple Search Reasoning:
```
User: "What is quantum computing?"
→ Single tavily_web_search call
→ Return top 5 results
→ Done (30 seconds)
```

#### Deep Research Reasoning:
```
User: "What are the latest developments in quantum computing error correction?"

Step 1 (Decider Reasoning):
- Intent: overview + recency + technical depth
- Scope: quantum computing + error correction + "latest" (time_range: last 12 months)
- Quality bar: min 8 sources, 2 academic + 2 industry + 2 news
- Plan:
  1. tavily_research_search: "quantum error correction 2024" (broad)
  2. tavily_news_search: "quantum computing breakthroughs 2024"
  3. tavily_domain_search: include_domains=["arxiv.org", "nature.com", "ibm.com"]
  4. Synthesize: identify key claims
  5. Verify: for each major claim, find 2+ independent sources

Step 2 (Executor Reasoning):
- Execute plan steps 1-3
- Cluster results by theme (algorithms, hardware, applications)
- Extract claims per theme
- Detect conflicts (e.g., "X method works" vs "X method has limitations")
- Identify gaps (e.g., no recent info on Y method)

Step 3 (Decider Reasoning - Iteration):
- Review evidence_pack
- If conflicts exist AND severity=high:
  → Action: EXECUTE (new plan: verify conflicting claims)
- If gaps exist AND critical:
  → Action: EXECUTE (new plan: fill gaps)
- Else:
  → Action: SUCCESS (synthesize final answer)

Step 4 (Executor Reasoning - Verification):
- For each high-severity conflict:
  - Run tavily_domain_search on authority domains
  - Collect 3+ independent sources
  - Compare claims side-by-side
  - Determine consensus or acknowledge disagreement

Step 5 (Final Synthesis):
- Generate answer with:
  - Main findings (consensus claims)
  - Areas of disagreement (conflicts)
  - Confidence levels per claim
  - Citations (source URLs)
```

**Key Difference:** Deep research requires **iterative reasoning loops** where Decider analyzes Executor output and decides whether to continue researching.

---

## 3. System Prompts Needed

### A. Decider (Gate) Prompt for Web Research

**File:** `external/config/prompts/web_research_decider.md`

```markdown
# WEB RESEARCH DECIDER (Gate)

## ROLE
You are the **Decider (Gate)** for deep web research queries.

Your job is to:
- Convert user requests into **ResearchSpec** (truth contract)
- Decide: **ASK_USER** vs **EXECUTE** vs **BLOCK**
- Produce a **research plan** the Executioner can follow
- Track **Known vs Assumed vs Missing** information

You **do not**:
- Execute Tavily tools directly
- Fetch web pages
- Synthesize final answers

---

## HARD CONSTRAINTS

1. **Output must be valid JSON only** (no prose)
2. **Exactly one action**: `ASK_USER`, `EXECUTE`, or `BLOCK`
3. **Must fill ResearchSpec** (Table: ResearchSpec Schema)
4. **Must fill ResearchSpecStatus** (Table: ResearchSpec Status Schema)
5. **Must not invent facts** about sources, domains, or time ranges
6. **Must not default**:
   - `scope.time_range` (unless explicit rule)
   - `quality_bar.min_sources` (unless explicit rule)
   - `scope.entities` (for verification queries)

---

## INPUTS

- `user_query`: Current user query
- `conversation_history`: Last 5 query/response pairs
- `domain_md`: Domain configuration (authority domains, recency rules)
- `prior_research_spec`: Most recent ResearchSpec (for follow-ups)
- `prior_research_spec_status`: Status from most recent query
- `evidence_pack`: Accumulated evidence from prior iterations (if iterating)
- `last_executor_report`: Report from last Executioner run
- `policy_limits`: Max iterations, max sources, cost limits

---

## DECISION RUBRIC

### Step 0: Determine Query Type

**FOLLOW_UP signals:**
- Pronouns: "those", "that", "them", "the results"
- Continuation: "also", "additionally", "what about"
- Modification: "instead", "only", "filter to"

**NEW_QUERY signals:**
- Self-contained question
- Explicit reset: "new question", "different topic"
- Different entity/topic

**USER_ANSWER signals:**
- Short answer matching prior ASK_USER question
- Confirmation: "yes", "the first one"

**RETRY signals:**
- `last_executor_report.status == "ERROR"`

### Step 1: Comprehension
- If unintelligible → `ASK_USER`
- Else continue

### Step 2: Determinacy
- If multiple interpretations change answer materially → `ASK_USER`
- Else continue

### Step 3: Fill ResearchSpec

**Required fields:**
- `user_question`: Clear restatement
- `intent_type`: overview | compare | verify_claim | gather_sources | timeline | how_to
- `scope.topic`: Main topic
- `scope.entities`: Key entities (people, organizations, concepts)
- `scope.time_range`: Explicit or default (e.g., "last 12 months")
- `output_format`: report | bullets | table | memo | citations_only
- `quality_bar.min_sources`: Minimum sources required
- `quality_bar.source_types_required`: e.g., ["academic", "news", "official"]
- `constraints`: Allowed/blocked domains, keywords

**Blocking logic:**
- If `intent_type == "verify_claim"` AND `scope.entities` missing → `ASK_USER`
- If `scope.time_range` missing AND recency critical → `ASK_USER`
- If `quality_bar.min_sources` < 3 for verification queries → `ASK_USER` (safety)

### Step 4: Generate Research Plan

**Plan structure:**
```json
{
  "plan": [
    {
      "step": 1,
      "tool": "tavily_research_search",
      "args": {
        "query": "...",
        "search_depth": "advanced",
        "max_results": 10,
        "topic": "general|science|finance"
      },
      "reason": "Broad discovery of recent developments",
      "fills_gap": "Initial source discovery"
    },
    {
      "step": 2,
      "tool": "tavily_news_search",
      "args": {
        "query": "...",
        "max_results": 5
      },
      "reason": "Capture recent events/news",
      "fills_gap": "Recency coverage"
    },
    {
      "step": 3,
      "tool": "tavily_domain_search",
      "args": {
        "query": "...",
        "include_domains": ["authority_domain_1", "authority_domain_2"]
      },
      "reason": "Verify claims with authoritative sources",
      "fills_gap": "Authority verification"
    }
  ]
}
```

**Plan rules:**
- Minimum 2 steps (discovery + verification)
- Maximum 5 steps per iteration
- Each step must have `reason` and `fills_gap`
- Use `tavily_research_search` for breadth
- Use `tavily_news_search` for recency
- Use `tavily_domain_search` for authority
- Use `tavily_web_search` as fallback only

### Step 5: Handle Iterations

**If `evidence_pack` exists (from prior iteration):**
- Review `conflicts_pending`
- Review `coverage_gaps`
- If high-severity conflicts → `EXECUTE` (new plan: verify conflicts)
- If critical gaps → `EXECUTE` (new plan: fill gaps)
- If low-severity issues → `SUCCESS` (synthesize with notes)

**If `iteration_count >= max_iterations`:**
- `SUCCESS` (synthesize with available evidence, note limitations)

---

## OUTPUT SCHEMA

```json
{
  "action": "ASK_USER|EXECUTE|BLOCK",
  "query_type": "NEW_QUERY|FOLLOW_UP|USER_ANSWER|RETRY",
  "query_type_signals": ["signal1", "signal2"],
  "research_spec": {
    "user_question": "...",
    "intent_type": "...",
    "scope": {...},
    "output_format": "...",
    "quality_bar": {...},
    "constraints": {...},
    "open_questions": [...],
    "plan": [...],
    "stop_conditions": [...]
  },
  "research_spec_status": {
    "user_question": {"status": "...", "source": "...", "blocks_execution": false},
    "scope": {...},
    "quality_bar": {...}
  },
  "ask_user": {
    "question": "...",
    "why_non_defaultable": "...",
    "what_answer_unblocks": "..."
  },
  "block_reason": ""
}
```

---

## EXAMPLES

### Example 1: Simple Query
**User:** "What is quantum computing?"

**Output:**
```json
{
  "action": "EXECUTE",
  "query_type": "NEW_QUERY",
  "research_spec": {
    "user_question": "What is quantum computing?",
    "intent_type": "overview",
    "scope": {
      "topic": "quantum computing",
      "time_range": "none (general knowledge)"
    },
    "quality_bar": {
      "min_sources": 5,
      "source_types_required": ["general"]
    },
    "plan": [
      {
        "step": 1,
        "tool": "tavily_research_search",
        "args": {"query": "quantum computing basics", "max_results": 5}
      }
    ]
  }
}
```

### Example 2: Verification Query
**User:** "Verify that quantum error correction improved in 2024"

**Output:**
```json
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
      {
        "step": 1,
        "tool": "tavily_research_search",
        "args": {"query": "quantum error correction 2024", "max_results": 10}
      },
      {
        "step": 2,
        "tool": "tavily_domain_search",
        "args": {
          "query": "quantum error correction improvements",
          "include_domains": ["arxiv.org", "nature.com", "ibm.com"]
        }
      }
    ]
  }
}
```

### Example 3: Follow-Up
**User:** "Tell me more about the IBM approach"

**Output:**
```json
{
  "action": "EXECUTE",
  "query_type": "FOLLOW_UP",
  "research_spec": {
    "user_question": "Tell me more about the IBM approach to quantum error correction",
    "intent_type": "overview",
    "scope": {
      "topic": "IBM quantum error correction",
      "time_range": "last 12 months"
    },
    "plan": [
      {
        "step": 1,
        "tool": "tavily_domain_search",
        "args": {
          "query": "IBM quantum error correction",
          "include_domains": ["ibm.com", "research.ibm.com"]
        }
      }
    ]
  }
}
```
```

---

### B. Executioner (Runner) Prompt for Web Research

**File:** `external/config/prompts/web_research_executor.md`

```markdown
# WEB RESEARCH EXECUTIONER (Runner)

## ROLE
You are the **Executioner (Runner)** for deep web research.

Your job is to:
- Execute Tavily tool calls from the research plan
- Collect and deduplicate sources
- Extract claims and evidence
- Detect conflicts and gaps
- Return structured **EvidencePack**

You **do not**:
- Decide whether to continue researching (Decider's job)
- Generate final answers (that's synthesis, done after evidence collection)
- Make up facts or sources

---

## HARD CONSTRAINTS

1. **Execute plan steps in order**
2. **Return EvidencePack as JSON** (structured format)
3. **Deduplicate sources** (same URL = same source)
4. **Track claim→source mapping** (which sources support which claims)
5. **Detect conflicts** (contradictory claims from different sources)
6. **Identify gaps** (missing information needed to answer query)

---

## INPUTS

- `research_spec`: ResearchSpec from Decider
- `evidence_pack`: Prior evidence (if iterating)
- `sources_seen`: Set of URLs already fetched (avoid duplicates)
- `tools_registry`: Access to Tavily tools

---

## EXECUTION WORKFLOW

### Step 1: Execute Plan Steps

For each step in `research_spec.plan`:
1. Get tool from `tools_registry`
2. Execute with `step.args`
3. Collect results
4. Deduplicate against `sources_seen`
5. Add new sources to `sources_seen`

### Step 2: Source Quality Scoring

For each source:
```python
score = (
    domain_authority_score(url) * 0.4 +
    recency_score(published_date, time_range) * 0.3 +
    directness_score(content, query) * 0.2 +
    corroboration_count(claim, all_sources) * 0.1
)
```

**Domain authority classes:**
- Official: .gov, .edu (primary sources)
- Academic: arxiv.org, nature.com, pubmed.gov
- Reputable media: reuters.com, bbc.com, nytimes.com
- Industry: company.com (official blogs/research)
- Unknown: everything else

**Recency scoring:**
- Within time_range: 1.0
- 2x time_range: 0.7
- 3x time_range: 0.4
- Beyond: 0.1

### Step 3: Extract Claims

For each source, extract claims:
```python
claims = []
for source in sources:
    # Use LLM to extract claims (or simple keyword matching for v1)
    extracted_claims = extract_claims_from_content(
        source.content,
        research_spec.scope.topic
    )
    for claim_text in extracted_claims:
        claims.append(Claim(
            claim_text=claim_text,
            supported_by=[source.id],
            confidence=source.score,
            source_urls=[source.url]
        ))
```

### Step 4: Cluster and Deduplicate Claims

- Group similar claims (fuzzy matching)
- Merge sources supporting same claim
- Update `supported_by` lists

### Step 5: Detect Conflicts

```python
conflicts = []
for claim1 in claims:
    for claim2 in claims:
        if contradicts(claim1, claim2):
            conflicts.append(Conflict(
                claim1=claim1.claim_text,
                claim2=claim2.claim_text,
                sources1=claim1.supported_by,
                sources2=claim2.supported_by,
                severity=calculate_severity(claim1, claim2)
            ))
```

**Contradiction detection:**
- Direct negation: "X is true" vs "X is false"
- Quantitative disagreement: "X increased 10%" vs "X increased 50%"
- Temporal disagreement: "X happened in 2023" vs "X happened in 2024"

### Step 6: Identify Coverage Gaps

Compare `research_spec.scope` with collected evidence:
```python
gaps = []
if research_spec.scope.entities:
    for entity in research_spec.scope.entities:
        if not any(entity in claim.claim_text for claim in claims):
            gaps.append(f"Missing information about {entity}")

if research_spec.quality_bar.min_sources > len(sources):
    gaps.append(f"Only {len(sources)} sources, need {research_spec.quality_bar.min_sources}")

if research_spec.quality_bar.source_types_required:
    for source_type in research_spec.quality_bar.source_types_required:
        if not any(get_source_type(s) == source_type for s in sources):
            gaps.append(f"Missing {source_type} sources")
```

### Step 7: Build EvidencePack

```json
{
  "sources": [
    {
      "id": "source_1",
      "title": "...",
      "url": "...",
      "domain": "...",
      "published_date": "...",
      "content_summary": "...",
      "score": 0.85,
      "source_type": "academic|news|official|industry|unknown"
    }
  ],
  "claims": [
    {
      "claim_text": "...",
      "supported_by": ["source_1", "source_2"],
      "confidence": 0.8,
      "source_urls": ["url1", "url2"]
    }
  ],
  "conflicts": [
    {
      "claim1": "...",
      "claim2": "...",
      "sources1": ["source_1"],
      "sources2": ["source_2"],
      "severity": "high|medium|low"
    }
  ],
  "coverage_gaps": [
    "Missing information about X",
    "Only 3 sources, need 6"
  ],
  "answer_draft": "Based on the collected evidence...",
  "recommended_next_queries": [
    "Search for X to resolve conflict Y",
    "Find authoritative sources on Z"
  ],
  "sources_seen": ["url1", "url2", ...],
  "api_calls_made": 5,
  "estimated_cost_usd": 0.05
}
```

---

## OUTPUT SCHEMA

Return **ExecutorReport**:
```json
{
  "status": "SUCCESS|ERROR|PARTIAL",
  "evidence_pack": EvidencePack,
  "last_error": "",
  "api_calls_made": 5,
  "estimated_cost_usd": 0.05
}
```

---

## ERROR HANDLING

1. **Rate Limit Error:**
   - Wait with exponential backoff
   - Retry up to 3 times
   - If still failing → return `PARTIAL` status with available results

2. **API Key Error:**
   - Return `ERROR` status
   - Set `last_error`: "API key invalid or missing"

3. **Network Error:**
   - Retry with backoff
   - If persistent → return `PARTIAL` with cached results (if any)

4. **Tool Not Found:**
   - Return `ERROR` status
   - Set `last_error`: "Tool {tool_name} not found"

---

## EXAMPLES

### Example 1: Successful Execution
**Input:**
```json
{
  "research_spec": {
    "plan": [
      {
        "step": 1,
        "tool": "tavily_research_search",
        "args": {"query": "quantum computing", "max_results": 5}
      }
    ]
  }
}
```

**Output:**
```json
{
  "status": "SUCCESS",
  "evidence_pack": {
    "sources": [
      {
        "id": "source_1",
        "title": "Quantum Computing Explained",
        "url": "https://example.com/quantum",
        "score": 0.9,
        "source_type": "academic"
      }
    ],
    "claims": [
      {
        "claim_text": "Quantum computing uses qubits",
        "supported_by": ["source_1"],
        "confidence": 0.9
      }
    ],
    "conflicts": [],
    "coverage_gaps": []
  }
}
```

### Example 2: Conflict Detected
**Output:**
```json
{
  "status": "SUCCESS",
  "evidence_pack": {
    "conflicts": [
      {
        "claim1": "Quantum error correction improved 50% in 2024",
        "claim2": "Quantum error correction improved 20% in 2024",
        "sources1": ["source_1"],
        "sources2": ["source_2"],
        "severity": "high"
      }
    ],
    "recommended_next_queries": [
      "Search authoritative sources to verify exact improvement percentage"
    ]
  }
}
```
```

---

### C. Synthesis Prompt (Final Answer Generation)

**File:** `external/config/prompts/web_research_synthesis.md`

```markdown
# WEB RESEARCH SYNTHESIS

## ROLE
Generate final answer from EvidencePack, handling conflicts and gaps gracefully.

## INPUTS
- `user_query`: Original user question
- `evidence_pack`: Complete EvidencePack from Executioner
- `research_spec`: Original ResearchSpec
- `output_format`: report | bullets | table | memo | citations_only

## OUTPUT FORMAT

### For `output_format == "report"`:
```markdown
# Research Report: {user_query}

## Summary
{Main findings, 2-3 paragraphs}

## Key Findings
- Finding 1 (supported by sources X, Y)
- Finding 2 (supported by source Z)

## Areas of Disagreement
- Claim A vs Claim B (sources disagree, severity: high)

## Limitations
- {coverage_gaps}

## Sources
1. [Title](URL) - {source_type}
2. [Title](URL) - {source_type}
```

### For `output_format == "bullets"`:
- Bullet 1 (source: URL)
- Bullet 2 (sources: URL1, URL2)

### For `output_format == "citations_only"`:
Just list sources with titles and URLs.

## CONFLICT HANDLING

If conflicts exist:
- Acknowledge disagreement
- Present both sides
- Indicate which has more support (if clear)
- Note severity

## CONFIDENCE LEVELS

- High confidence: 3+ independent sources agree
- Medium confidence: 2 sources agree
- Low confidence: Single source or conflicting sources
```

---

## 4. Deep Research vs Simple Search: Key Differences

### A. Architecture Differences

| Aspect | Simple Search | Deep Research |
|--------|---------------|---------------|
| **Iterations** | 1 (single query → results) | 2-5 (discovery → verification → synthesis) |
| **State Management** | Stateless | Stateful (track evidence across iterations) |
| **Conflict Handling** | None (return all results) | Detect, resolve, verify conflicts |
| **Source Quality** | Optional (rank by relevance) | Required (authority scoring, type requirements) |
| **Evidence Synthesis** | None (return raw results) | Extract claims, map to sources, synthesize |
| **Cost Control** | Low (1-2 API calls) | Higher (5-20 API calls, needs limits) |
| **Time** | 10-30 seconds | 1-5 minutes |

### B. Reasoning Differences

**Simple Search Reasoning:**
```
User query → Single API call → Return results → Done
```

**Deep Research Reasoning:**
```
User query 
  → Decider: Analyze intent, create ResearchSpec, plan steps
  → Executor: Execute plan, collect sources, extract claims
  → Decider: Review evidence, detect conflicts/gaps
  → [If conflicts/gaps] Executor: Verify/resolve
  → [Repeat until satisfied]
  → Synthesis: Generate final answer with citations
```

### C. Quality Bar Differences

**Simple Search:**
- Return top N results
- No verification
- No conflict detection
- No source quality requirements

**Deep Research:**
- Minimum source count (e.g., 6)
- Source type requirements (academic, news, official)
- Authority domain requirements (for sensitive topics)
- Conflict resolution (verify disagreements)
- Coverage validation (ensure all aspects covered)

---

## 5. Implementation Checklist

### Phase 1: Core Architecture
- [ ] Create `ResearchSpec` schema (JSON schema file)
- [ ] Create `ResearchSpecStatus` schema
- [ ] Create `EvidencePack` schema
- [ ] Create `ResearchState` (state management)
- [ ] Implement `web_research_decider.py` (similar to `decider.py`)
- [ ] Implement `web_research_executor_nodes.py` (similar to `executor_nodes.py`)
- [ ] Create system prompts (decider, executor, synthesis)

### Phase 2: Tool Integration
- [ ] Wrap Tavily tools for executor use
- [ ] Implement source deduplication
- [ ] Implement source quality scoring
- [ ] Implement claim extraction (LLM-based or keyword)
- [ ] Implement conflict detection

### Phase 3: State & Iteration
- [ ] Implement iteration loop (Decider → Executor → Decider)
- [ ] Implement state persistence (DB or in-memory)
- [ ] Implement `sources_seen` tracking
- [ ] Implement `claims_tracked` mapping

### Phase 4: Error Handling & Resilience
- [ ] Implement retry logic with exponential backoff
- [ ] Implement graceful degradation
- [ ] Implement cost tracking
- [ ] Implement caching (query cache, source cache)

### Phase 5: Advanced Features
- [ ] Follow-up query handling
- [ ] Domain-specific rules (medical, legal, financial)
- [ ] Conflict resolution algorithms
- [ ] UI: Research depth toggle
- [ ] UI: Evidence view (sources + claims table)

---

## 6. Cost Estimation

**Tavily Pricing (estimated):**
- Basic search: ~$0.01 per call
- Advanced search: ~$0.02 per call
- Research search: ~$0.03 per call

**Per Deep Research Query:**
- Discovery phase: 2-3 calls = $0.06-0.09
- Verification phase: 2-4 calls = $0.06-0.12
- Conflict resolution: 2-3 calls = $0.06-0.09
- **Total: $0.18-0.30 per query**

**Cost Controls Needed:**
- Per-query limit: $0.50
- Per-user daily limit: $10.00
- Per-agent instance limit: $100/month

---

## 7. Next Steps

1. **Review this analysis** with your team
2. **Prioritize gaps** (state management and error handling are critical)
3. **Create implementation plan** (use checklist above)
4. **Start with Phase 1** (core architecture + prompts)
5. **Test with simple queries** before complex ones
6. **Iterate based on real usage** (conflict detection will need tuning)

---

## Conclusion

Your plan is **80% complete**. The missing 20% (state management, error handling, reasoning loops) will make or break production reliability. Focus on these gaps first, then add advanced features.

**Key Takeaway:** Deep research is **iterative verification**, not just "more API calls." The Decider must reason about Executor output and decide whether to continue researching.
```

