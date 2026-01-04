# Financial News Research Agent for Bankers - Example Configuration

This document provides example prompts and domain configuration for a web research agent that searches financial news, regulators, and other sources to find relevant information about companies for bankers.

---

## 1. Domain Content (domain_file.md)

This is the domain markdown file you would upload when creating the agent instance:

```markdown
# Financial News Research Agent - Domain Configuration

## Authority Domains

Primary authoritative sources for financial and regulatory information:

- **Regulatory Bodies:**
  - sec.gov (Securities and Exchange Commission)
  - federalreserve.gov (Federal Reserve System)
  - fdic.gov (Federal Deposit Insurance Corporation)
  - occ.treas.gov (Office of the Comptroller of the Currency)
  - finra.org (Financial Industry Regulatory Authority)
  - cftc.gov (Commodity Futures Trading Commission)
  - treasury.gov (U.S. Department of the Treasury)

- **Financial News & Analysis:**
  - bloomberg.com
  - reuters.com
  - wsj.com (Wall Street Journal)
  - ft.com (Financial Times)
  - marketwatch.com
  - cnbc.com
  - financialtimes.com

- **Company Filings & Disclosures:**
  - sec.gov/edgar (EDGAR database)
  - investor relations pages of major companies

- **Industry Analysis:**
  - sifma.org (Securities Industry and Financial Markets Association)
  - aba.com (American Bankers Association)
  - bis.org (Bank for International Settlements)

## Search Scope

**Allowed Domains (Priority):**
- sec.gov
- federalreserve.gov
- fdic.gov
- bloomberg.com
- reuters.com
- wsj.com
- ft.com
- marketwatch.com
- cnbc.com
- finra.org
- treasury.gov

**Blocked Domains:**
- reddit.com
- twitter.com
- facebook.com
- personal blogs
- social media platforms
- unverified financial advice sites

## Research Depth Settings

**Default Research Depth:** Standard (2 iterations, 6-20 sources)

**Quality Bar:**
- Minimum sources: 6
- Maximum sources: 20
- Minimum authority sources (regulatory/official): 2
- Source types required: ["regulatory", "news", "official"]

**Time Range Default:**
- For "recent" queries: Last 12 months
- For company-specific queries: Last 24 months (to capture filing cycles)
- For regulatory queries: All time (regulations are historical)

## Search Strategy

1. **Primary Search:** Regulatory filings (SEC EDGAR, company disclosures)
2. **Secondary Search:** Financial news (Bloomberg, Reuters, WSJ)
3. **Tertiary Search:** Regulatory announcements (SEC, Fed, FDIC)
4. **Verification:** Cross-reference multiple authoritative sources

## Key Topics for Bankers

- Company financial performance
- Regulatory compliance issues
- SEC filings (10-K, 10-Q, 8-K)
- Enforcement actions
- Market analysis
- Credit risk indicators
- Industry trends
- Management changes
- M&A activity
- Litigation and settlements

## Output Format Preferences

- **Default:** Report format with executive summary
- **For quick queries:** Bullet points
- **For detailed analysis:** Full report with citations
- **For regulatory queries:** Memo format

## Special Instructions

1. **Company Identification:**
   - Always verify company name and ticker symbol
   - Check for subsidiaries and related entities
   - Note any name changes or mergers

2. **Regulatory Focus:**
   - Prioritize SEC filings for public companies
   - Check for enforcement actions
   - Look for regulatory warnings or sanctions

3. **Risk Indicators:**
   - Flag any negative news or regulatory issues
   - Highlight financial distress signals
   - Note any litigation or settlements

4. **Source Attribution:**
   - Always cite specific SEC filings (e.g., "SEC Form 10-K filed on [date]")
   - Include publication dates for news articles
   - Note source authority level (regulatory > news > industry)

5. **Confidence Levels:**
   - High: Multiple regulatory sources or major news outlets agree
   - Medium: Single authoritative source or multiple news sources
   - Low: Single unverified source or conflicting information
```

---

## 2. Example Agent Configuration

When creating the agent in the admin panel:

**Agent Name:** `Financial News Research Agent for Bankers`

**Description:** `Searches financial news, regulatory filings, and official sources to provide comprehensive company intelligence for banking professionals. Focuses on SEC filings, regulatory compliance, financial performance, and risk indicators.`

**Agent Type:** `external` (web research)

**Model:** `claude-sonnet-4-20250514` (recommended for complex financial analysis)

**Tavily API Key:** (your Tavily API key)

**Allowed Domains:** 
```
sec.gov, federalreserve.gov, fdic.gov, bloomberg.com, reuters.com, wsj.com, ft.com, marketwatch.com, cnbc.com, finra.org, treasury.gov, occ.treas.gov, cftc.gov
```

**Blocked Domains:**
```
reddit.com, twitter.com, facebook.com, x.com
```

**Default Research Depth:** `standard` (2 iterations, 6-20 sources)

**Domain File:** Upload the `domain_file.md` content above

---

## 3. Example Customized Prompts (Per-Agent Overrides)

These are examples of how you might customize the prompts for this specific agent. You would add these as agent-specific prompt overrides in the "Prompts" tab when editing the agent.

### 3.1 Web Research Decider (Customized)

You can customize the decider prompt to add financial-specific guidance:

```markdown
# WEB RESEARCH DECIDER - Financial News Research Agent

[Include all standard decider content, but add this section:]

## FINANCIAL RESEARCH SPECIFIC GUIDANCE

### Company Identification
- Always extract company name, ticker symbol, and any subsidiaries
- Verify company identity before proceeding
- If company name is ambiguous → ASK_USER for clarification

### Regulatory Priority
- For public companies: ALWAYS prioritize SEC filings (10-K, 10-Q, 8-K)
- For banks: Include FDIC, OCC, and Federal Reserve sources
- For broker-dealers: Include FINRA sources

### Time Range Logic (Financial)
- **SEC filings:** Check last 2 years (filing cycles)
- **Recent news:** Last 12 months
- **Regulatory actions:** All time (historical context matters)
- **Market events:** Last 6 months for "recent" queries

### Quality Bar (Financial)
- Minimum 2 regulatory/official sources (SEC, Fed, FDIC)
- Minimum 4 news sources from authoritative outlets
- Total minimum: 6 sources
- Maximum: 20 sources

### Research Plan Structure (Financial)
1. **Step 1:** SEC EDGAR search (company filings)
2. **Step 2:** Financial news search (Bloomberg, Reuters, WSJ)
3. **Step 3:** Regulatory announcements (SEC, Fed, FDIC)
4. **Step 4:** Industry analysis (if needed)

### Risk Indicators to Flag
- SEC enforcement actions
- Regulatory warnings
- Financial distress signals
- Litigation or settlements
- Management changes
- M&A activity

[Rest of standard decider prompt...]
```

### 3.2 Web Research Synthesis (Customized)

```markdown
# WEB RESEARCH SYNTHESIS - Financial News Research Agent

[Include all standard synthesis content, but add this section:]

## FINANCIAL REPORT STRUCTURE

### Executive Summary (Always Include)
- Company name and ticker
- Key findings (2-3 sentences)
- Risk level (Low/Medium/High)
- Date of research

### Report Sections (Standard)
1. **Company Overview**
   - Basic information
   - Business description
   - Recent developments

2. **Financial Performance**
   - Revenue trends
   - Profitability
   - Key financial metrics
   - SEC filing highlights

3. **Regulatory Compliance**
   - SEC filings status
   - Enforcement actions (if any)
   - Regulatory warnings
   - Compliance issues

4. **Risk Assessment**
   - Financial risks
   - Regulatory risks
   - Operational risks
   - Market risks

5. **Recent News & Events**
   - Major news (last 12 months)
   - Management changes
   - M&A activity
   - Market developments

6. **Sources & Citations**
   - SEC filings (with form numbers and dates)
   - News articles (with publication dates)
   - Regulatory announcements

### Risk Indicators Format

**High Risk:**
- SEC enforcement actions
- Regulatory sanctions
- Financial distress (bankruptcy filings, defaults)
- Major litigation

**Medium Risk:**
- Regulatory warnings
- Negative financial trends
- Management turnover
- Industry headwinds

**Low Risk:**
- Stable financial performance
- No regulatory issues
- Positive industry trends

[Rest of standard synthesis prompt...]
```

### 3.3 Web Research Claim Extraction (Customized)

```markdown
# WEB RESEARCH CLAIM EXTRACTION - Financial News Research Agent

[Include all standard claim extraction content, but add this section:]

## FINANCIAL CLAIM CATEGORIES

### Categories for Financial Research
- **Financial Performance:** Revenue, profit, margins, growth
- **Regulatory Compliance:** SEC filings, enforcement, warnings
- **Risk Indicators:** Financial distress, litigation, defaults
- **Market Activity:** Stock performance, analyst ratings, M&A
- **Management:** Executive changes, board actions
- **Operations:** Business changes, product launches, market expansion

### Financial Claim Examples

**Good Financial Claims:**
- "Company XYZ reported revenue of $1.2B in Q3 2024, up 15% YoY"
- "SEC filed enforcement action against Company XYZ on [date] for [violation]"
- "Company XYZ filed Form 10-K on [date] reporting net loss of $50M"
- "Bloomberg reported that Company XYZ is exploring sale of [division]"

**Bad Financial Claims:**
- "Company XYZ is doing well" (too vague)
- "Some people think Company XYZ has issues" (opinion, not fact)
- "What is Company XYZ's financial status?" (question, not claim)

### Source Authority for Financial Claims

**High Authority (High Confidence):**
- SEC filings (10-K, 10-Q, 8-K)
- Federal Reserve announcements
- FDIC reports
- Official regulatory documents

**Medium Authority (Medium Confidence):**
- Bloomberg, Reuters, WSJ
- Financial Times
- MarketWatch, CNBC

**Lower Authority (Lower Confidence):**
- Industry blogs
- Unverified financial sites
- Social media

[Rest of standard claim extraction prompt...]
```

---

## 4. Example User Queries and Expected Behavior

### Example 1: Company Overview Query

**User Query:** "Find recent news and regulatory information about Apple Inc. for a credit risk assessment"

**Expected Research Spec:**
```json
{
  "user_question": "Find recent news and regulatory information about Apple Inc. for a credit risk assessment",
  "intent_type": "overview",
  "scope": {
    "topic": "Apple Inc. financial performance, regulatory compliance, and risk indicators",
    "entities": ["Apple Inc.", "AAPL"],
    "time_range": "last 24 months"
  },
  "quality_bar": {
    "min_sources": 8,
    "max_sources": 20,
    "min_authority_sources": 2,
    "source_types_required": ["regulatory", "news", "official"]
  },
  "plan": [
    {
      "step": 1,
      "tool": "tavily_domain_search",
      "args": {
        "query": "Apple Inc AAPL SEC filings 10-K 10-Q",
        "include_domains": ["sec.gov"]
      },
      "reason": "Get official SEC filings for financial performance",
      "fills_gap": "Regulatory compliance and financial data"
    },
    {
      "step": 2,
      "tool": "tavily_news_search",
      "args": {
        "query": "Apple Inc financial news credit risk 2024",
        "max_results": 10
      },
      "reason": "Capture recent financial news and market analysis",
      "fills_gap": "Recent developments and market sentiment"
    },
    {
      "step": 3,
      "tool": "tavily_domain_search",
      "args": {
        "query": "Apple Inc regulatory enforcement SEC",
        "include_domains": ["sec.gov", "federalreserve.gov"]
      },
      "reason": "Check for regulatory issues or enforcement actions",
      "fills_gap": "Regulatory risk assessment"
    }
  ]
}
```

### Example 2: Verification Query

**User Query:** "Verify that Tesla Inc. filed their 10-K on time in 2024"

**Expected Research Spec:**
```json
{
  "user_question": "Verify that Tesla Inc. filed their 10-K on time in 2024",
  "intent_type": "verify_claim",
  "scope": {
    "topic": "Tesla Inc. SEC Form 10-K filing",
    "entities": ["Tesla Inc.", "TSLA", "Form 10-K"],
    "time_range": "2024"
  },
  "quality_bar": {
    "min_sources": 6,
    "max_sources": 15,
    "min_authority_sources": 2,
    "source_types_required": ["regulatory", "official"]
  },
  "plan": [
    {
      "step": 1,
      "tool": "tavily_domain_search",
      "args": {
        "query": "Tesla TSLA Form 10-K 2024 filing date",
        "include_domains": ["sec.gov"]
      },
      "reason": "Verify filing date from official SEC source",
      "fills_gap": "Official filing verification"
    },
    {
      "step": 2,
      "tool": "tavily_news_search",
      "args": {
        "query": "Tesla 10-K filing deadline 2024",
        "max_results": 5
      },
      "reason": "Cross-reference with news reports",
      "fills_gap": "Additional verification"
    }
  ]
}
```

### Example 3: Risk Assessment Query

**User Query:** "What are the regulatory risks for JPMorgan Chase in the last year?"

**Expected Research Spec:**
```json
{
  "user_question": "What are the regulatory risks for JPMorgan Chase in the last year?",
  "intent_type": "overview",
  "scope": {
    "topic": "JPMorgan Chase regulatory risks and compliance issues",
    "entities": ["JPMorgan Chase", "JPM"],
    "time_range": "last 12 months"
  },
  "quality_bar": {
    "min_sources": 8,
    "max_sources": 20,
    "min_authority_sources": 3,
    "source_types_required": ["regulatory", "news", "official"]
  },
  "plan": [
    {
      "step": 1,
      "tool": "tavily_domain_search",
      "args": {
        "query": "JPMorgan Chase regulatory enforcement SEC FDIC",
        "include_domains": ["sec.gov", "fdic.gov", "federalreserve.gov", "occ.treas.gov"]
      },
      "reason": "Check for regulatory enforcement actions and warnings",
      "fills_gap": "Official regulatory risk assessment"
    },
    {
      "step": 2,
      "tool": "tavily_news_search",
      "args": {
        "query": "JPMorgan Chase regulatory issues compliance 2024",
        "max_results": 10
      },
      "reason": "Capture news about regulatory issues",
      "fills_gap": "Media coverage of regulatory risks"
    },
    {
      "step": 3,
      "tool": "tavily_research_search",
      "args": {
        "query": "JPMorgan Chase regulatory risk analysis banking",
        "max_results": 5,
        "search_depth": "advanced"
      },
      "reason": "Find in-depth analysis of regulatory risks",
      "fills_gap": "Comprehensive risk assessment"
    }
  ]
}
```

---

## 5. Expected Output Format

### Example Output for Company Overview Query

```markdown
# Company Research Report: Apple Inc. (AAPL)

**Research Date:** [Current Date]  
**Risk Level:** Low  
**Sources:** 12 authoritative sources

## Executive Summary

Apple Inc. (AAPL) demonstrates strong financial performance with consistent revenue growth and robust regulatory compliance. No significant regulatory issues or enforcement actions identified in the review period. The company maintains high credit quality with stable cash flows and strong market position.

## Company Overview

Apple Inc. is a multinational technology company that designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.

**Recent Developments:**
- [Key development from news sources]
- [Major announcement from SEC filings]

## Financial Performance

**Key Metrics (from SEC Form 10-K):**
- Revenue: $[amount] (fiscal year [year])
- Net Income: $[amount]
- Total Assets: $[amount]
- Cash and Cash Equivalents: $[amount]

**Trends:**
- Revenue growth: [X]% YoY
- Profitability: [stable/improving/declining]

**Sources:**
- SEC Form 10-K filed on [date] (Source: sec.gov/edgar)
- Bloomberg: "[Article Title]" published on [date]

## Regulatory Compliance

**SEC Filings Status:**
- ✅ All required filings submitted on time
- ✅ No enforcement actions
- ✅ No regulatory warnings

**Recent Filings:**
- Form 10-K: Filed on [date]
- Form 10-Q (Q3): Filed on [date]
- Form 8-K: [Recent significant events]

**Sources:**
- SEC EDGAR database (sec.gov/edgar)
- SEC press releases

## Risk Assessment

**Financial Risks:** Low
- Strong cash position
- Consistent profitability
- Diversified revenue streams

**Regulatory Risks:** Low
- Full compliance with SEC requirements
- No enforcement actions
- Transparent disclosure practices

**Operational Risks:** Low-Medium
- [Any operational concerns from sources]

**Market Risks:** Medium
- [Market-related risks from sources]

## Recent News & Events (Last 12 Months)

1. **[Date]:** [Major news event] (Source: Bloomberg)
2. **[Date]:** [Management change] (Source: Reuters)
3. **[Date]:** [Product launch] (Source: WSJ)

## Sources & Citations

### Regulatory Sources
1. SEC Form 10-K - Apple Inc. (filed [date]) - sec.gov/edgar
2. SEC Form 10-Q - Apple Inc. Q3 2024 (filed [date]) - sec.gov/edgar
3. Federal Reserve Economic Data - [if applicable]

### News Sources
1. Bloomberg: "[Article Title]" - Published [date] - bloomberg.com/...
2. Reuters: "[Article Title]" - Published [date] - reuters.com/...
3. Wall Street Journal: "[Article Title]" - Published [date] - wsj.com/...

### Industry Analysis
1. [Industry report source]

---

**Confidence Level:** High  
**Research Completeness:** Comprehensive  
**Open Questions:** [Any gaps or areas needing clarification]
```

---

## 6. Setup Instructions

1. **Create the Agent:**
   - Go to Admin Panel → Manage Agents → External
   - Click "Create New Agent"
   - Fill in the agent details (name, description, model)
   - Upload the `domain_file.md` content
   - Set allowed/blocked domains
   - Set default research depth to "standard"

2. **Customize Prompts (Optional):**
   - Edit the agent
   - Go to "Prompts" tab
   - For each prompt you want to customize:
     - Click "Create Override"
     - Paste the customized prompt content
     - Save

3. **Test the Agent:**
   - Go to Agent Chat
   - Select your new agent
   - Try queries like:
     - "Find recent news about Microsoft for credit risk assessment"
     - "What are the regulatory risks for Bank of America?"
     - "Verify that Amazon filed their 10-K on time in 2024"

---

## Notes

- The agent will automatically use the domain configuration to prioritize regulatory sources
- Evidence pack will show sources, claims, conflicts, and gaps
- The synthesis will format output specifically for banking professionals
- All sources are cited with URLs and dates
- Risk indicators are clearly flagged

This configuration provides a comprehensive financial news research agent tailored for banking professionals who need company intelligence for credit risk assessment, compliance monitoring, and due diligence.

