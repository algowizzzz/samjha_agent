# PM TLDR: Agent Types & AI Workflows

---

## 1. Structured Data Agent

### What It Is
An AI assistant that queries your **internal structured data** (CSV, Parquet, databases) using natural language. Think of it as a smart analyst who can answer questions about your data without needing SQL skills.

### Business Value
- **Self-serve analytics**: Business users ask questions in plain English → get table/chart results
- **No data exposure**: Data stays internal, never leaves your environment
- **Audit trail**: Every query is logged with the SQL generated

### How It Works (Simplified)
```
User: "What were our top 5 products by revenue last quarter?"
    ↓
Decider (AI): Understands intent, creates execution plan
    ↓
Executor: Generates SQL, runs query safely, returns results
    ↓
User sees: Table with top 5 products + revenue numbers
```

### Key Capabilities
| Feature | Description |
|---------|-------------|
| Natural Language Queries | Ask questions like talking to an analyst |
| SQL Generation | Auto-generates safe, validated SQL |
| Follow-up Support | "Now filter that to Canada only" works |
| Clarification | Agent asks questions when query is ambiguous |
| Domain-Aware | Understands your business terms (e.g., "churn" = specific calculation) |

### Best For
- Internal reporting & dashboards
- Ad-hoc data exploration
- Empowering non-technical users to query data
- Replacing repetitive SQL requests to data teams

---

## 2. Web Research Agent

### What It Is
An AI assistant that performs **real-time web research** using the Tavily search API. It searches, extracts evidence, identifies conflicts, and synthesizes answers with citations.

### Business Value
- **Market research on-demand**: Get synthesized research with sources in minutes
- **Competitive intelligence**: Track competitors, industry news, regulatory changes
- **Fact-checked answers**: AI synthesizes from multiple sources, flags conflicts
- **Source transparency**: Every claim links to its source

### How It Works (Simplified)
```
User: "What are the latest AI regulations in the EU?"
    ↓
Decider (AI): Plans research scope & search strategy
    ↓
Executor: Searches web → Extracts claims → Detects conflicts → Synthesizes
    ↓
User sees: Answer with citations + list of sources + any conflicting info
```

### Key Capabilities
| Feature | Description |
|---------|-------------|
| Multi-source Research | Searches multiple web sources for comprehensive coverage |
| Evidence Extraction | Pulls out specific claims from each source |
| Conflict Detection | Flags when sources disagree |
| Citation Support | Every claim attributed to its source |
| Configurable Depth | Quick (fast), Standard, or Deep (comprehensive) research |
| Domain Filtering | Restrict to trusted domains (or block specific sites) |

### Best For
- Market & competitive research
- Due diligence research
- Regulatory monitoring
- News & trend analysis
- Any question requiring current, external information

---

## 3. AI Workflows (Bulk Document Analysis)

### What It Is
An **automated document processing pipeline** that takes multiple documents through a series of AI-powered analysis steps. Think of it as a production line for document intelligence.

### Business Value
- **Scale document processing**: Process hundreds of documents with consistent AI analysis
- **Standardized extraction**: Same prompts applied across all docs → comparable outputs
- **Flexible pipelines**: Configure ingestion → analysis steps → export format
- **Audit trail**: Every step logged with inputs/outputs

### How It Works (Simplified)
```
Documents (PDFs, etc.)
    ↓
[Ingestion Profile] - How to parse/prepare documents
    ↓
[Chain] - Series of AI analysis steps (prompts)
  Step 1: Extract key obligations
  Step 2: Identify risks
  Step 3: Summarize findings
    ↓
[Export Profile] - Output format (Markdown, Excel, JSON)
    ↓
Structured output for each document
```

### Core Concepts
| Component | What It Does |
|-----------|-------------|
| **Workflow** | The full end-to-end pipeline (groups everything) |
| **Ingestion Profile** | How documents are converted/prepared (PDF, images, etc.) |
| **Chain** | Series of analysis steps (each step = one AI prompt) |
| **Chain Step** | Single AI analysis task (e.g., "extract obligations") |
| **Export Profile** | Output format configuration (MD, Excel, JSON) |
| **Execution** | One run of the workflow on a set of documents |

### Best For
- Contract analysis at scale
- Regulatory document review
- Due diligence document processing
- Any repetitive document analysis task
- Compliance/audit document review

### Key Differences from Agents

| Aspect | Structured/Web Agents | AI Workflows |
|--------|----------------------|--------------|
| Input | Natural language questions | Documents (PDFs, etc.) |
| Output | Answers with data/sources | Structured extractions |
| Mode | Interactive Q&A | Batch processing |
| Use Case | Ad-hoc queries | Standardized document review |

---

## Summary: When to Use What

| Scenario | Best Option |
|----------|-------------|
| "What were our sales last month?" | **Structured Data Agent** |
| "What are competitors saying about AI?" | **Web Research Agent** |
| "Analyze these 50 contracts for risks" | **AI Workflows** |
| User needs quick internal data answer | **Structured Data Agent** |
| User needs current external research | **Web Research Agent** |
| Need consistent analysis across many docs | **AI Workflows** |

---

*Document generated for PM reference. For technical architecture, see `STRUCTURED_WEB_AGENT_OVERVIEW.md` and `HOW_TO_CREATE_CHAINS_WORKFLOWS.md`.*

