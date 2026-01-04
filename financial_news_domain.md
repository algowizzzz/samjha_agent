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

