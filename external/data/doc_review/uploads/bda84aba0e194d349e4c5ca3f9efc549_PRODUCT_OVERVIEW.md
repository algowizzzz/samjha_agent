# Text-to-SQL Analytics Service
**Stateless Natural Language Query Interface for Enterprise Data**

---

## Executive Summary

A configurable, stateless service that enables business users across any department to query structured data using natural language—eliminating the technical barrier between business questions and data insights.

**Core Value:** Any team can deploy their own instance with their dataset in under 30 minutes, reducing query turnaround time from hours to seconds.

---

## Problem Statement

### Current State Across the FI
- **Risk Teams:** Analysts wait 2-4 hours for IT to write SQL for CCR exposure queries
- **Finance Teams:** Controllers manually export P&L data to Excel for ad-hoc analysis
- **Trading Desks:** Traders can't self-serve position breakdowns by book/desk/product
- **Compliance Teams:** Auditors request the same regulatory reports repeatedly
- **Operations:** Settlement teams lack quick access to failed trade summaries

### Business Impact
- **Productivity Loss:** 60-80% of queries are variations of 10 standard questions
- **Delayed Decisions:** Time-sensitive insights (breach identification, P&L swings) arrive too late
- **Resource Inefficiency:** Technical analysts spend time on repetitive queries instead of analysis
- **Data Silos:** Each team builds custom Excel macros that break with schema changes

---

## Solution: Stateless Analytics Service

### Architecture
```
User Query (Natural Language)
    ↓
Stateless Service (no database, no persistence)
    ↓
Configuration (form.json) + Data (Parquet files)
    ↓
SQL Generation (LLM or rule-based)
    ↓
In-Memory Execution (DuckDB)
    ↓
Result + Optional Commentary
```

**Key Principle:** Service owns **zero state**. Teams own their data and configuration.

---

## How It Works

### 1. **Team Onboards Their Dataset**
```bash
# Finance Team Example
input/
  └── form.json           # Configure: tables, columns, business vocab
  └── data/
      ├── pnl.parquet     # Daily P&L by desk
      └── bookings.parquet # Trade bookings
```

**Configuration defines:**
- Table schemas (columns and relationships)
- Business vocabulary (e.g., "top movers" → SQL for daily P&L change)
- Rule-based queries for common questions
- Optional LLM prompts for ad-hoc questions

### 2. **Users Query in Natural Language**
```bash
# Example: Trading Desk
$ python json_sql_copilot.py --form input/form.json \
  --q "Show me top 10 trades by notional for Desk A today"

# Example: Finance Controller
$ python json_sql_copilot.py --form input/form.json \
  --q "What desks had P&L swings over $5M yesterday?"

# Example: Compliance
$ python json_sql_copilot.py --form input/form.json \
  --q "How many failed trades by counterparty this week?"
```

### 3. **Service Returns Data + Insights**
- **SQL Generated:** Transparent query shown to user
- **Results:** JSON/CSV export ready
- **Commentary (Optional):** LLM-generated bullet points explaining key insights

---

## Key Features

### ✅ **Stateless & Lightweight**
- No database installation or maintenance
- No user authentication/authorization (delegates to existing FI systems)
- Runs on any laptop/server with Python
- Scales horizontally (spin up multiple instances)

### ✅ **Configuration-Driven**
- Each team creates `form.json` for their dataset
- Schema documentation embedded in config (self-documenting)
- Business vocabulary maps domain terms to SQL logic
- No code changes needed to onboard new datasets

### ✅ **Production-Ready for Large Datasets**
- **Parquet format:** Handles 75M+ rows efficiently
- **Zero-copy reads:** DuckDB native Parquet support
- **Partitioned data:** Date-based partitioning for faster queries
- **Two-stage optimization:** Filter extraction reduces LLM costs 30-50%

### ✅ **Dual Query Modes**
1. **Rule-Based (No LLM):** Pre-defined queries for common questions (fast, free)
2. **LLM-Powered:** Ad-hoc natural language queries (flexible, requires API key)

### ✅ **Enterprise Safety**
- **SQL Guardrails:** SELECT-only, no writes, LIMIT clamping
- **Column ownership validation:** Prevents schema errors
- **Timeout protection:** Queries auto-kill after 30 seconds
- **Audit trail:** All generated SQL logged

---

## Use Cases Across the FI

| Department | Dataset | Sample Query |
|------------|---------|--------------|
| **CCR Risk** | Exposure limits, trades | "Which counterparties breached limits this week?" |
| **Market Risk** | VaR, stress scenarios | "Show me top 10 risk factor contributors to VaR" |
| **Finance** | P&L, bookings | "What was Desk A's P&L for FX swaps in Q1?" |
| **Trading** | Positions, greeks | "Give me delta exposure by currency pair" |
| **Compliance** | Failed trades, breaches | "List all failed trades with notional > $10M" |
| **Operations** | Settlements, breaks | "How many settlement failures by custodian?" |
| **Treasury** | Funding, liquidity | "What's our largest liquidity drawdown counterparty?" |

**Key Insight:** Same service, different configurations. Each team deploys independently.

---

## Deployment Models

### Model 1: **Team Self-Service (Recommended)**
- Each team runs their own instance
- Team controls data refresh schedule (daily Parquet exports)
- Team owns configuration (form.json)
- Zero IT dependency after initial setup

### Model 2: **Centralized Service**
- IT hosts service with multiple configurations
- Teams submit `form.json` + Parquet files
- Service routes queries to appropriate config
- Requires governance for schema changes

### Model 3: **Hybrid (Best of Both)**
- IT provides base Docker image + documentation
- Teams deploy to their own environments
- Central repository for reusable configs (templates)

---

## Implementation Roadmap

### Phase 1: Pilot (1-2 Weeks)
- **Team:** CCR Risk (already configured)
- **Goal:** Validate on real queries, collect feedback
- **Success Metric:** 80% of common queries answered without IT escalation

### Phase 2: Expansion (1 Month)
- **Teams:** Finance (P&L), Market Risk (VaR), Trading (Positions)
- **Deliverable:** Template configs for common FI datasets
- **Goal:** 3-5 teams live

### Phase 3: Enterprise Rollout (Ongoing)
- **Self-Service Portal:** Teams request onboarding, IT provides guidance
- **Config Library:** Reusable templates (e.g., "Trade data standard config")
- **Training:** 30-min workshops for BAs on config creation

---

## Technical Requirements

### Infrastructure
- **Compute:** 2 CPU, 4GB RAM (handles 10M+ rows)
- **Storage:** Teams provide Parquet files (no service storage)
- **Network:** Outbound HTTPS for LLM API (optional, firewall-friendly)

### Dependencies
- Python 3.9+
- 7 lightweight packages (DuckDB, Pandas, OpenAI, PyArrow, etc.)
- Optional: OpenAI API key (or any OpenAI-compatible LLM)

### Data Format
- **Required:** Parquet files (columnar, compressed)
- **Conversion:** Simple Python script converts JSON/CSV → Parquet
- **Size:** Handles 100M+ rows per table on standard hardware

---

## Governance & Compliance

### Data Ownership
- **Teams own data:** Service never stores or persists data
- **Ephemeral processing:** Data loaded into memory, query executed, memory cleared
- **No data movement:** Parquet files stay in team's environment

### Security
- **No authentication built-in:** Delegates to FI's existing IAM
- **SQL injection prevention:** Parameterized queries, guardrails
- **Audit logging:** All queries + generated SQL logged to file/SIEM

### Compliance-Friendly
- **No PII storage:** Stateless = no retention
- **GDPR/CCPA ready:** No persistent user data
- **SOX-compliant:** Read-only access, full audit trail

---

## Cost Analysis

### Per-Team TCO (Annual)
| Component | Cost | Notes |
|-----------|------|-------|
| Infrastructure | $500-1,000 | VM/container hosting |
| LLM API (optional) | $0-5,000 | ~$0.10 per 1,000 queries (GPT-4o-mini) |
| Data Storage | $0 | Teams use existing data lake |
| Maintenance | 5-10 hours | Occasional config updates |

**ROI Calculation:**
- **Before:** 100 queries/month × 2 hours analyst time × $100/hour = **$20,000/month**
- **After:** 100 queries/month × 5 minutes × $100/hour = **$833/month**
- **Savings:** ~$230,000/year per team

---

## Success Metrics

### Adoption Metrics
- **Teams onboarded:** Target 10 teams in 6 months
- **Query volume:** 500+ queries/month across all teams
- **Self-service rate:** 75% of queries answered without IT escalation

### Efficiency Metrics
- **Time-to-insight:** <30 seconds (vs. 2-4 hours previously)
- **Query accuracy:** 90%+ correct SQL on first attempt
- **User satisfaction:** >4.5/5 in feedback surveys

### Business Impact
- **Analyst productivity:** 60% reduction in time spent on repetitive queries
- **Decision latency:** Real-time breach identification (was: next day)
- **IT load reduction:** 80% fewer ad-hoc SQL requests

---

## Competitive Advantages

vs. **Traditional BI Tools (Tableau, Power BI):**
- ✅ No need to build dashboards upfront
- ✅ Handles ad-hoc questions instantly
- ✅ Natural language, not drag-and-drop

vs. **Data Warehouse Query UIs:**
- ✅ No SQL knowledge required
- ✅ Business vocabulary built-in
- ✅ Stateless = zero DBA overhead

vs. **Custom Internal Tools:**
- ✅ Configurable across datasets (one codebase, many use cases)
- ✅ Production-ready for large data (75M+ rows)
- ✅ Open architecture (not vendor-locked)

---

## Next Steps

### For Product Managers
1. **Identify pilot teams:** Which teams have high query volumes + clean Parquet exports?
2. **Define success criteria:** What does "success" look like for first 3 teams?
3. **Budget approval:** $10-20K for first year (infra + LLM API)

### For Engineering
1. **Infrastructure setup:** Provision VMs/containers for pilot teams
2. **Config templates:** Create reusable form.json templates for common datasets
3. **Monitoring:** Set up query logging + performance dashboards

### For Business Analysts
1. **Learn configuration:** 2-hour workshop on form.json creation
2. **Document vocabulary:** Map business terms → SQL for your dataset
3. **Test queries:** Validate 10 most common questions

---

## Contact & Resources

**GitHub Repository:** https://github.com/algowizzzz/TextToSQL  
**Documentation:** See README.md for technical setup  
**Support:** [Your team's contact info here]

---

**Version:** 1.0  
**Last Updated:** October 2025  
**Maintained By:** [Your team name]

