# Web Search Agent Configuration Guide

## Overview

Web search agents use a **domain configuration file** (similar to `domain_md` for structured agents) that controls:
- **Which URLs/domains to search** (e.g., SEC-only, academic-only)
- **Research depth** (quick vs. standard vs. deep)
- **Source quality requirements** (authority domains, minimum sources)
- **Search preferences** (time ranges, topics, excluded domains)
- **API configuration** (Tavily API key per instance)

---

## 1. Domain Configuration File Structure

**File Format:** Markdown (`.md`) stored in `domain_content` column (same as structured agents)

**Location:** Created via Admin Panel → Stored in DB → Loaded at runtime

### Example: SEC-Only Financial Agent

```markdown
# SEC Financial Research Agent Configuration

## Domain Type
financial_regulatory

## Authority Domains (REQUIRED)
These domains are prioritized and may be required for certain queries.

### Primary Authorities
- sec.gov (U.S. Securities and Exchange Commission)
- federalreserve.gov (Federal Reserve)
- treasury.gov (U.S. Treasury)
- fdic.gov (FDIC)
- finra.org (FINRA)

### Secondary Authorities
- nasdaq.com (official filings)
- nyse.com (official filings)
- sec.gov/edgar (EDGAR database)

## Allowed Domains (OPTIONAL)
If specified, ONLY search these domains. Leave empty to search all domains.

- sec.gov
- sec.gov/edgar
- federalreserve.gov
- treasury.gov
- fdic.gov
- finra.org
- nasdaq.com
- nyse.com

## Blocked Domains (OPTIONAL)
Never search these domains, even if they appear in results.

- reddit.com
- twitter.com
- personal-blogs.com
- unverified-news.com

## Research Depth Settings

### Quick Research (Default)
- max_iterations: 1
- min_sources: 3
- max_sources: 10
- max_api_calls: 5
- search_depth: "basic"

### Standard Research
- max_iterations: 2
- min_sources: 6
- max_sources: 20
- max_api_calls: 10
- search_depth: "advanced"

### Deep Research
- max_iterations: 4
- min_sources: 10
- max_sources: 50
- max_api_calls: 20
- search_depth: "advanced"
- require_authority_sources: true
- min_authority_sources: 3

## Source Quality Requirements

### Minimum Source Types
For verification queries, require at least:
- 2 official sources (gov domains)
- 1 industry source (regulated entity)
- 1 news source (reputable media)

### Authority Scoring Rules
- .gov domains: score = 1.0
- .edu domains: score = 0.9
- Regulated entities (sec.gov/edgar): score = 0.95
- Reputable media: score = 0.7
- Industry blogs: score = 0.5
- Unknown domains: score = 0.3

## Time Range Defaults

### Financial Regulations
- Default: "last 12 months"
- Critical updates: "last 3 months"
- Historical: "last 5 years"

### Market Data
- Default: "last 30 days"
- Real-time queries: "last 7 days"

## Topic Categories

### Financial Topics
- securities_regulation
- banking_regulation
- market_analysis
- company_filings
- enforcement_actions

## Search Tool Preferences

### Preferred Tools (in order)
1. tavily_domain_search (for authority domains)
2. tavily_research_search (for comprehensive research)
3. tavily_news_search (for recent updates)
4. tavily_web_search (fallback only)

## Conflict Resolution Rules

### High-Severity Conflicts
- Require 3+ independent authority sources
- Must include at least 1 official source (.gov)
- If unresolved after 2 iterations → ASK_USER

### Medium-Severity Conflicts
- Require 2+ independent sources
- Note disagreement in final answer
- Continue with majority consensus

## Cost Limits

### Per Query
- max_cost_usd: 0.50
- warn_at_usd: 0.30

### Per Agent Instance (Monthly)
- max_cost_usd: 100.00
- alert_at_usd: 75.00

## Output Format Defaults

### Financial Reports
- Default: "report" (structured report with sections)
- Alternative: "table" (for comparative data)

### Citations Style
- Format: "[Title](URL) - {source_type}"
- Include: published_date, domain, source_type
```

---

## 2. Admin Panel Configuration

### A. Creating a Web Search Agent

**Route:** `/admin` → "Agent Instances" → "Create New Agent"

**Step 1: Agent Type Selection**
```
Select Agent Type:
┌─────────────────────────────────────────┐
│ [Dropdown]                              │
│ ✅ Structured Data (Available)          │
│ ✅ Web Search (Available) ← NEW         │
│ ⏳ Unstructured Data (Coming Soon)       │
└─────────────────────────────────────────┘
```

**Step 2: Basic Information**
```
Agent Name *: [SEC Financial Research Agent]
Description: [Research SEC filings, regulations, and financial data]
LLM Model *: [Claude 3 Sonnet (Recommended)] [Dropdown]
```

**Step 3: Domain Configuration File**
```
Domain File *: [Choose File] [sec_financial_domain.md]
               ↑ Upload your .md configuration file
```

**Step 4: API Configuration**
```
Tavily API Key *: [********************] [Show/Hide]
                  ↑ Per-instance API key (stored encrypted)
                  
API Key Source:
○ Use global default
● Use instance-specific key ← Recommended for isolation
```

**Step 5: Search Scope Configuration**
```
Search Scope:
┌─────────────────────────────────────────┐
│ ○ Search all domains (default)          │
│ ● Restrict to specific domains          │
│                                         │
│ Allowed Domains (one per line):        │
│ ┌─────────────────────────────────────┐│
│ │ sec.gov                              ││
│ │ federalreserve.gov                   ││
│ │ treasury.gov                         ││
│ │                                       ││
│ └─────────────────────────────────────┘│
│                                         │
│ Blocked Domains (one per line):        │
│ ┌─────────────────────────────────────┐│
│ │ reddit.com                           ││
│ │ twitter.com                          ││
│ │                                       ││
│ └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

**Step 6: Research Depth Default**
```
Default Research Depth:
┌─────────────────────────────────────────┐
│ ○ Quick (1 iteration, 3-10 sources)     │
│ ● Standard (2 iterations, 6-20 sources) │
│ ○ Deep (4 iterations, 10-50 sources)    │
└─────────────────────────────────────────┘
```

**Step 7: Cost Limits**
```
Cost Limits:
┌─────────────────────────────────────────┐
│ Per Query Limit (USD): [0.50]          │
│ Monthly Limit (USD): [100.00]          │
│ Alert at (USD): [75.00]                │
└─────────────────────────────────────────┘
```

**Step 8: Advanced Settings (Collapsible)**
```
[+] Advanced Settings

Source Quality Requirements:
┌─────────────────────────────────────────┐
│ Minimum Sources: [6]                    │
│ Minimum Authority Sources: [2]           │
│                                         │
│ Required Source Types:                  │
│ ☑ Official (.gov)                       │
│ ☑ Academic (.edu)                       │
│ ☐ Industry                              │
│ ☐ News                                  │
└─────────────────────────────────────────┘

Time Range Defaults:
┌─────────────────────────────────────────┐
│ Default: [last 12 months]              │
│ Critical Updates: [last 3 months]        │
│ Historical: [last 5 years]              │
└─────────────────────────────────────────┘
```

---

## 3. All Control Variables

### A. Domain Configuration File Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `domain_type` | string | Domain category | `financial_regulatory`, `medical`, `legal` |
| `authority_domains` | array | Prioritized domains | `["sec.gov", "federalreserve.gov"]` |
| `allowed_domains` | array | **ONLY** search these (restrictive) | `["sec.gov"]` (SEC-only agent) |
| `blocked_domains` | array | Never search these | `["reddit.com", "twitter.com"]` |
| `research_depth.quick` | object | Quick research settings | `{max_iterations: 1, min_sources: 3}` |
| `research_depth.standard` | object | Standard research settings | `{max_iterations: 2, min_sources: 6}` |
| `research_depth.deep` | object | Deep research settings | `{max_iterations: 4, min_sources: 10}` |
| `source_quality.min_source_types` | object | Required source types | `{official: 2, academic: 1}` |
| `authority_scoring_rules` | object | Domain scoring weights | `{".gov": 1.0, ".edu": 0.9}` |
| `time_range_defaults` | object | Default time ranges | `{financial: "last 12 months"}` |
| `topic_categories` | array | Domain-specific topics | `["securities_regulation", "banking"]` |
| `preferred_tools` | array | Tool priority order | `["tavily_domain_search", "tavily_research_search"]` |
| `conflict_resolution.high_severity` | object | High conflict rules | `{min_sources: 3, require_authority: true}` |
| `cost_limits.per_query` | number | Max cost per query (USD) | `0.50` |
| `cost_limits.monthly` | number | Max cost per month (USD) | `100.00` |
| `output_format_default` | string | Default output format | `"report"` |

### B. Agent Instance Variables (DB)

| Variable | Type | Description | Control Location |
|----------|------|-------------|------------------|
| `agent_id` | string | Unique identifier | Auto-generated from name |
| `name` | string | Display name | Admin Panel → Name field |
| `agent_type` | string | Agent type | Admin Panel → Type dropdown |
| `description` | string | Description | Admin Panel → Description field |
| `domain_file` | string | Domain file filename | Admin Panel → Domain File upload |
| `domain_content` | text | Domain config markdown | Admin Panel → Domain File content |
| `tavily_api_key` | string | API key (encrypted) | Admin Panel → API Key field |
| `model` | string | LLM model | Admin Panel → Model dropdown |
| `search_scope.allowed_domains` | array | Allowed domains override | Admin Panel → Search Scope |
| `search_scope.blocked_domains` | array | Blocked domains override | Admin Panel → Search Scope |
| `default_research_depth` | string | Default depth | Admin Panel → Research Depth |
| `cost_limits.per_query` | number | Per-query limit | Admin Panel → Cost Limits |
| `cost_limits.monthly` | number | Monthly limit | Admin Panel → Cost Limits |

---

## 4. Example: Creating SEC-Only Agent

### Step-by-Step

**1. Create Domain Configuration File**

Create `sec_financial_domain.md`:

```markdown
# SEC Financial Research Agent

## Domain Type
financial_regulatory

## Allowed Domains (SEC-ONLY)
- sec.gov
- sec.gov/edgar
- federalreserve.gov
- treasury.gov
- fdic.gov
- finra.org

## Blocked Domains
- reddit.com
- twitter.com
- personal-blogs.com
- unverified-news.com

## Research Depth Settings

### Quick Research
- max_iterations: 1
- min_sources: 3
- max_sources: 10
- search_depth: "basic"

### Standard Research
- max_iterations: 2
- min_sources: 6
- max_sources: 20
- search_depth: "advanced"

### Deep Research
- max_iterations: 4
- min_sources: 10
- max_sources: 50
- search_depth: "advanced"
- require_authority_sources: true
- min_authority_sources: 3

## Source Quality Requirements
- Minimum 2 official sources (.gov)
- Minimum 1 industry source (regulated entity)

## Time Range Defaults
- Default: "last 12 months"
- Critical updates: "last 3 months"

## Preferred Tools
1. tavily_domain_search (for SEC.gov)
2. tavily_research_search (for comprehensive)
3. tavily_news_search (for recent filings)
```

**2. Admin Panel Setup**

1. Go to `/admin` → "Agent Instances"
2. Click "Create New Agent"
3. Select "Web Search" as agent type
4. Fill in:
   - **Name:** "SEC Financial Research Agent"
   - **Description:** "Research SEC filings, regulations, and financial compliance data"
   - **Model:** Claude 3 Sonnet
   - **Domain File:** Upload `sec_financial_domain.md`
   - **API Key:** Enter your Tavily API key
   - **Search Scope:** Select "Restrict to specific domains"
   - **Allowed Domains:** Paste list (sec.gov, federalreserve.gov, etc.)
   - **Default Research Depth:** Standard
   - **Cost Limits:** Per query $0.50, Monthly $100

**3. Result**

You now have a **SEC-only web search agent** that:
- ✅ Only searches SEC, Federal Reserve, Treasury domains
- ✅ Requires 2+ official sources for verification
- ✅ Uses standard research depth (2 iterations, 6-20 sources)
- ✅ Blocks Reddit, Twitter, unverified sources
- ✅ Defaults to "last 12 months" for financial queries

---

## 5. How Domain Config Controls Research Depth

### Quick Research (User Toggle: "Quick")
```
Domain Config: research_depth.quick
→ max_iterations: 1
→ min_sources: 3
→ max_api_calls: 5
→ search_depth: "basic"

Flow:
User query → Decider → Executor (1 pass) → Done
Time: ~30 seconds
Cost: ~$0.05
```

### Standard Research (User Toggle: "Standard")
```
Domain Config: research_depth.standard
→ max_iterations: 2
→ min_sources: 6
→ max_api_calls: 10
→ search_depth: "advanced"

Flow:
User query → Decider → Executor → Decider (review) → Executor (verify) → Done
Time: ~1-2 minutes
Cost: ~$0.15
```

### Deep Research (User Toggle: "Deep")
```
Domain Config: research_depth.deep
→ max_iterations: 4
→ min_sources: 10
→ max_api_calls: 20
→ search_depth: "advanced"
→ require_authority_sources: true

Flow:
User query → Decider → Executor → Decider (review conflicts) → Executor (verify) → Decider (review gaps) → Executor (fill gaps) → Synthesis
Time: ~3-5 minutes
Cost: ~$0.30
```

**User can override** via UI toggle, but domain config sets **defaults and limits**.

---

## 6. Admin Panel UI Mockup

### Create Agent Modal (Web Search Type)

```
┌─────────────────────────────────────────────────────────────┐
│ Create New Agent                                    [✕ Close]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Agent Type: [Web Search ▼]                                 │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Basic Information                                        ││
│ ├─────────────────────────────────────────────────────────┤│
│ │ Agent Name *: [SEC Financial Research Agent        ]    ││
│ │ Description: [Research SEC filings and regulations]     ││
│ │ LLM Model *: [Claude 3 Sonnet ▼]                       ││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Domain Configuration                                    ││
│ ├─────────────────────────────────────────────────────────┤│
│ │ Domain File *: [Choose File] [sec_financial_domain.md] ││
│ │                                                         ││
│ │ Preview: [Show Domain Config]                           ││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ API Configuration                                        ││
│ ├─────────────────────────────────────────────────────────┤│
│ │ Tavily API Key *: [********************] [Show]        ││
│ │                                                         ││
│ │ API Key Source:                                         ││
│ │ ○ Use global default                                    ││
│ │ ● Use instance-specific key                             ││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Search Scope                                             ││
│ ├─────────────────────────────────────────────────────────┤│
│ │ ○ Search all domains (default)                          ││
│ │ ● Restrict to specific domains                          ││
│ │                                                         ││
│ │ Allowed Domains (one per line):                         ││
│ │ ┌─────────────────────────────────────────────────────┐││
│ │ │ sec.gov                                              │││
│ │ │ federalreserve.gov                                   │││
│ │ │                                                       │││
│ │ └─────────────────────────────────────────────────────┘││
│ │                                                         ││
│ │ Blocked Domains (one per line):                         ││
│ │ ┌─────────────────────────────────────────────────────┐││
│ │ │ reddit.com                                           │││
│ │ │                                                       │││
│ │ └─────────────────────────────────────────────────────┘││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Research Settings                                       ││
│ ├─────────────────────────────────────────────────────────┤│
│ │ Default Research Depth:                                 ││
│ │ ○ Quick    ● Standard    ○ Deep                        ││
│ │                                                         ││
│ │ [Advanced Settings ▼]                                   ││
│ │   Minimum Sources: [6]                                  ││
│ │   Minimum Authority Sources: [2]                        ││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Cost Limits                                             ││
│ ├─────────────────────────────────────────────────────────┤│
│ │ Per Query Limit (USD): [0.50]                           ││
│ │ Monthly Limit (USD): [100.00]                           ││
│ │ Alert at (USD): [75.00]                                 ││
│ └─────────────────────────────────────────────────────────┘│
│                                                             │
│                                    [Cancel]  [Create Agent] │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Runtime Behavior

### How Domain Config is Used

**1. Decider (Gate) reads domain config:**
```python
# In web_research_decider.py
domain_md = load_domain_config(agent_id)  # Loads from domain_content

# Extract settings
allowed_domains = parse_allowed_domains(domain_md)
authority_domains = parse_authority_domains(domain_md)
research_depth = parse_research_depth(domain_md, user_selected_depth)
min_sources = research_depth["min_sources"]
```

**2. ResearchSpec creation:**
```python
research_spec = {
    "scope": {
        "topic": user_query_topic,
        "time_range": domain_md["time_range_defaults"]["default"],
        "entities": extract_entities(user_query)
    },
    "constraints": {
        "allowed_domains": allowed_domains,  # From domain config
        "blocked_domains": blocked_domains,  # From domain config
    },
    "quality_bar": {
        "min_sources": research_depth["min_sources"],  # From domain config
        "source_types_required": domain_md["source_quality"]["min_source_types"]
    },
    "plan": generate_plan(domain_md["preferred_tools"])  # Tool priority from config
}
```

**3. Executor (Runner) enforces constraints:**
```python
# In web_research_executor_nodes.py
for step in research_spec.plan:
    # Check if domain is allowed
    if step.tool == "tavily_domain_search":
        allowed = research_spec.constraints.allowed_domains
        if allowed and domain not in allowed:
            skip_step()  # Don't search blocked domains
    
    # Score sources using authority rules
    source_score = calculate_authority_score(
        source.url,
        domain_md["authority_scoring_rules"]
    )
```

---

## 8. Summary: What You Control

### ✅ Domain Configuration File (`domain_content`)
- Authority domains (prioritized)
- Allowed domains (restrictive: SEC-only, academic-only, etc.)
- Blocked domains (never search)
- Research depth settings (quick/standard/deep)
- Source quality requirements
- Time range defaults
- Tool preferences
- Conflict resolution rules
- Cost limits

### ✅ Admin Panel Fields
- Agent name & description
- LLM model selection
- Domain file upload
- Tavily API key (per instance)
- Search scope (allowed/blocked domains override)
- Default research depth
- Cost limits (per query, monthly)
- Advanced settings (min sources, authority requirements)

### ✅ Runtime Overrides (User UI)
- Research depth toggle (Quick/Standard/Deep)
- Time range selection
- Output format (report/bullets/table)

---

## 9. Example Configurations

### A. Academic Research Agent
```markdown
## Allowed Domains
- arxiv.org
- pubmed.gov
- nature.com
- science.org
- scholar.google.com

## Research Depth
- Deep research default
- min_sources: 10
- require_authority_sources: true (academic only)
```

### B. Medical Research Agent
```markdown
## Authority Domains
- pubmed.gov
- nih.gov
- who.int
- cdc.gov

## Blocked Domains
- personal-blogs.com
- unverified-news.com

## Time Range
- Default: "last 5 years" (medical research is slower)
```

### C. Legal Research Agent
```markdown
## Allowed Domains
- supremecourt.gov
- law.cornell.edu
- justia.com
- findlaw.com

## Research Depth
- Deep research default
- min_sources: 8
- require_authority_sources: true
```

---

## 10. Next Steps

1. **Create domain config templates** for common use cases (SEC, academic, medical, legal)
2. **Update admin panel** to support web search agent type
3. **Add API key encryption** for secure storage
4. **Implement domain config parser** (similar to structured agent domain_md parser)
5. **Add validation** (ensure allowed_domains/blocked_domains don't conflict)

---

## Key Takeaway

**Domain configuration file = Your control panel for web search agents**

Just like structured agents use `domain_md` to define data schemas and business rules, web search agents use `domain_content` to define:
- **Where to search** (allowed/blocked domains)
- **How deep to research** (iterations, sources, API calls)
- **What quality to require** (authority sources, source types)
- **How to resolve conflicts** (verification rules)

This gives you **full control** over each agent instance without code changes.

