# BMO Financials Data Summary

## Overview
The `bmo_financials` folder contains **7 Excel files** that are **BMO Financial Group quarterly supplement reports**. These are detailed financial disclosure documents that provide supplementary information to the main quarterly reports.

## Files Inventory

| Filename | Size | Likely Period |
|----------|------|---------------|
| Suppq323.xlsx | 590,832 bytes | Q3 2023 Supplement |
| Suppq323 (1).xlsx | 590,832 bytes | Q3 2023 Supplement (duplicate) |
| Suppq324.xlsx | 601,678 bytes | Q4 2023 Supplement |
| Suppq325 (1).xlsx | 729,242 bytes | Q1 2024 Supplement |
| Suppq423.xlsx | 592,400 bytes | Q4 2023 Supplement (alternate?) |
| Suppq424.xlsx | 767,327 bytes | Q4 2024 Supplement |
| Suppq425.xlsx | 733,169 bytes | Q1 2025 Supplement |

## File Structure

Each file contains **approximately 39+ sheets** organized as:

1. **Cover** - Cover page
2. **Index** - Table of contents showing all pages and sections
3. **Page 1** through **Page 37+** - Actual financial data pages

## Content Types (from Index Sheet)

Based on the Index sheet analysis, these files contain:

### 1. **Notes to Users**
   - Usage instructions
   - Methodology and definitions

### 2. **Financial Highlights**
   - Income Statement Information
   - Reported Profitability Measures
   - Adjusted Profitability Measures
   - Growth Rates
   - Balance Sheet Information
   - Capital Measures

### 3. **Financial Statements**
   - Statement of Comprehensive Income
   - Statement of Changes in Equity
   - Various balance sheet components

### 4. **Segmented Information**
   - Operating Group Results
   - Return on Equity by Operating Segment
   - Operating segment profitability

### 5. **Regulatory & Capital Information**
   - Capital allocation rates
   - Risk weighted assets
   - Regulatory capital requirements

### 6. **Credit Risk Information** (Pages 24-33)
   - Credit Risk Financial Measures
   - Provision for Credit Losses Segmented Information
   - Credit risk schedules

### 7. **Assets & Goodwill**
   - Goodwill and Intangible Assets
   - Assets Under Administration and Management
   - Unrealized Gains/Losses

## Key Characteristics

- **Format**: Financial report-style Excel files with formatted pages
- **Currency**: All amounts in millions of Canadian dollars (unless indicated)
- **Status**: Unaudited supplementary information
- **Purpose**: Designed to improve understanding of financial performance
- **Companion Documents**: Used with main quarterly reports and annual reports

## Data Challenges

1. **Unnamed columns**: Most sheets have "Unnamed" columns (likely formatted tables)
2. **Multi-page structure**: Data spread across many sheets
3. **Formatted layout**: Not traditional tabular data, more like printed reports
4. **Headers and footnotes**: Mixed with actual data values

## Potential Use Cases

- Extract financial metrics across quarters
- Compare performance between periods (Q3 2023 vs Q4 2023, etc.)
- Analyze trends in profitability measures
- Review credit risk measures over time
- Track capital allocation changes
- Analyze operating segment performance

## Recommendations for Analysis

1. **Identify key data sheets**: Focus on pages with actual financial tables
2. **Extract numerical data**: Look for sheets with structured financial metrics
3. **Cross-quarter comparison**: Compare same metrics across different files
4. **Date normalization**: Each file represents a different quarter/period
5. **Sheet mapping**: Use Index sheet to understand what each page contains

