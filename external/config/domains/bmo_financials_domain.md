# Domain: BMO Financials

## 1) Domain Identity

- **domain_key**: bmo_financials
- **description**: BMO Financial Group quarterly supplement reports containing detailed financial disclosures, regulatory information, and performance metrics across multiple reporting periods.

---

## 2) File Structure

### File Naming Convention
Files follow the pattern `Suppq{period}.xlsx` where:
- `Supp` = Supplement
- `q` = Quarter identifier
- Numbers indicate period (e.g., 323 = Q3 2023, 425 = Q1 2025)

### Available Files
- **Suppq323.xlsx** - Q3 2023 Supplement (July-September 2023)
- **Suppq324.xlsx** - Q4 2023 Supplement (October-December 2023)
- **Suppq325 (1).xlsx** - Q1 2024 Supplement (January-March 2024)
- **Suppq423.xlsx** - Q4 2023 Supplement (alternate version)
- **Suppq424.xlsx** - Q4 2024 Supplement (October-December 2024)
- **Suppq425.xlsx** - Q1 2025 Supplement (January-March 2025)

### Sheet Structure
Each file contains **39+ sheets** organized as:
1. **Cover** - Title page
2. **Index** - Table of contents (use this to understand what each page contains)
3. **Page 1** through **Page 37+** - Financial data pages

**Important**: Always check the **Index** sheet first to understand what data is on each page.

---

## 3) Data Characteristics

### Currency
- All amounts are in **millions of Canadian dollars** (CAD) unless indicated otherwise
- Look for footnotes or headers indicating currency changes

### Format Challenges
1. **Unnamed columns**: Sheets use "Unnamed: 0", "Unnamed: 1", etc. - data is in formatted report style
2. **Mixed content**: Headers, footnotes, and data values are mixed together
3. **Report layout**: Not traditional tabular data - formatted like printed financial reports
4. **Multiple sheets**: Data is spread across many sheets (use Index to navigate)

### Data Quality
- Status: **Unaudited** supplementary information
- Use in conjunction with main quarterly reports for complete picture

---

## 4) Content Types & Locations

### A) Notes to Users (Page 1)
- Usage instructions
- Methodology and definitions
- Important disclaimers

### B) Financial Highlights (Pages 5-6)
- **Income Statement Information**
- **Reported Profitability Measures** (ROE, ROA, efficiency ratio)
- **Adjusted Profitability Measures**
- **Growth Rates** (revenue, net income, etc.)
- **Balance Sheet Information**
- **Capital Measures** (regulatory capital, CET1 ratio)

### C) Financial Statements (Pages 21-23)
- Statement of Comprehensive Income
- Statement of Changes in Equity
- Balance Sheet components

### D) Segmented Information (Pages 5, 21)
- **Operating Group Results**
  - Canadian Personal & Commercial Banking
  - U.S. Personal & Commercial Banking
  - BMO Wealth Management
  - BMO Capital Markets
- **Return on Equity by Operating Segment**
- Operating segment profitability metrics

### E) Regulatory & Capital Information (Pages 6, 24+)
- **Capital allocation rates** (e.g., 11.0% of risk-weighted assets)
- **Risk-weighted assets** (RWA)
- **Regulatory capital requirements**
- **CET1 ratio** (Common Equity Tier 1)
- Capital adequacy measures

### F) Credit Risk Information (Pages 24-33)
- **Credit Risk Financial Measures**
- **Provision for Credit Losses** (PCL) segmented information
- Credit risk schedules
- Non-performing loans
- Allowance for credit losses

### G) Assets & Goodwill (Page 23)
- Goodwill and Intangible Assets
- Assets Under Administration and Management (AUA/AUM)
- Unrealized Gains (Losses) on Fair Value through Other Comprehensive Income Securities

---

## 5) Query Strategy

### When Querying This Domain

1. **Start with Index**: Always read the Index sheet to understand what's in each page
2. **Identify period**: Note which quarter/period the file represents
3. **Locate relevant pages**: Use Index to find pages containing the metrics you need
4. **Cross-period comparison**: Compare same metrics across different quarterly files
5. **Extract numerical data**: Look for cells with actual financial values, not headers

### Common Query Patterns

#### Time-Based Queries
- "Compare profitability between Q3 2023 and Q4 2023"
- "Show me capital ratios across all quarters"
- "What was the trend in credit losses from 2023 to 2025?"

#### Metric Extraction
- "What was the ROE in Q4 2023?"
- "Show me net income by operating segment for Q1 2024"
- "What is the CET1 ratio in the most recent quarter?"

#### Segment Analysis
- "Compare performance across operating segments"
- "Which segment had highest revenue growth?"
- "Show me credit risk by segment"

#### Trend Analysis
- "How has provision for credit losses changed over time?"
- "What is the trend in capital allocation rate?"
- "Show me assets under management growth"

---

## 6) Key Metrics to Track

### Profitability Metrics
- **Net Income** (reported and adjusted)
- **Revenue** (total revenue by segment)
- **Return on Equity (ROE)**
- **Return on Assets (ROA)**
- **Efficiency Ratio**

### Capital Metrics
- **Common Equity Tier 1 (CET1) Ratio**
- **Risk-Weighted Assets (RWA)**
- **Capital allocation rate**
- **Regulatory capital requirements**

### Credit Risk Metrics
- **Provision for Credit Losses (PCL)**
- **Non-performing loans (NPL)**
- **Allowance for credit losses**
- **Credit risk by segment**

### Balance Sheet Metrics
- **Total Assets**
- **Total Liabilities**
- **Total Equity**
- **Assets Under Administration/Management (AUA/AUM)**

### Growth Metrics
- **Revenue growth** (quarter-over-quarter, year-over-year)
- **Net income growth**
- **Loan growth**
- **Deposit growth**

---

## 7) File Identification

### Period Mapping
- **Suppq323** = Q3 2023 (Fiscal quarter ending ~July-September 2023)
- **Suppq324** = Q4 2023 (Fiscal quarter ending ~October-December 2023)
- **Suppq325** = Q1 2024 (Fiscal quarter ending ~January-March 2024)
- **Suppq423** = Q4 2023 (Alternate or duplicate)
- **Suppq424** = Q4 2024 (Fiscal quarter ending ~October-December 2024)
- **Suppq425** = Q1 2025 (Fiscal quarter ending ~January-March 2025)

### Fiscal Year Reference
- BMO's fiscal year typically ends October 31
- Q1 = November-January
- Q2 = February-April
- Q3 = May-July
- Q4 = August-October

**Note**: Verify exact dates by checking file contents, as fiscal periods may vary.

---

## 8) Data Extraction Guidelines

### When Reading Sheets

1. **Skip Cover page**: Usually just title, no data
2. **Use Index strategically**: Read Index to identify which Page contains the metric you need
3. **Focus on numerical cells**: Look for cells with actual financial values
4. **Check for footnotes**: Important qualifiers may be in footnotes
5. **Account for formatting**: Headers and data may be in non-standard positions

### Example Extraction Process

1. Query: "What was net income in Q3 2023?"
   - Identify file: `Suppq323.xlsx`
   - Check Index sheet to find "Income Statement" or "Financial Highlights"
   - Navigate to indicated Page (likely Page 5 or 21)
   - Extract numerical value for "Net Income"
   - Note currency and period

2. Query: "Compare ROE across quarters"
   - Identify all files (Suppq323, Suppq324, etc.)
   - For each file, check Index for "Profitability Measures" or "ROE"
   - Extract ROE value from each quarter
   - Present as comparison table

---

## 9) Best Practices

1. **Always read Index first** when exploring a new file
2. **Verify period** - check file name and content to confirm which quarter
3. **Note currency** - assume CAD millions unless stated otherwise
4. **Check footnotes** - important qualifiers and definitions
5. **Cross-reference** - use Index to find related information across pages
6. **Handle duplicates** - Suppq323.xlsx and Suppq323 (1).xlsx are likely identical

---

## 10) Use Cases

### Financial Analysis
- Extract key profitability metrics
- Compare performance across periods
- Analyze trends in financial ratios

### Regulatory Compliance
- Track capital adequacy ratios
- Monitor credit risk measures
- Review regulatory capital allocation

### Segment Performance
- Compare operating group results
- Analyze segment profitability
- Identify growth drivers

### Risk Management
- Track provision for credit losses
- Monitor non-performing loans
- Analyze credit risk trends

---

*This domain guide helps the Excel Agent understand BMO Financials supplement reports structure and extract meaningful insights from the quarterly financial disclosures.*

