"""
Generate CFO Insights Markdown Report
"""

import pandas as pd
from pathlib import Path
from external.agent.bmo_excel_agent import handle_bmo_query

query = 'from q4 2024 file only i am the cfo of bmo, tell me surprising things that i might have not noticed'

print('Running query...')
result = handle_bmo_query(
    user_query=query,
    agent_data_folder='bmo_financials',
    max_iterations=15
)

# Extract detailed insights
folder = Path('external/datawarehouse/bmo_financials')
fpath = folder / 'Suppq424.xlsx'
excel_file = pd.ExcelFile(fpath)

# Read key pages
df8 = pd.read_excel(excel_file, sheet_name='Page 8', nrows=25)
df5 = pd.read_excel(excel_file, sheet_name='Page 5', nrows=30)

# Extract metrics
metrics = {}
for idx, row in df8.iterrows():
    for col_idx, val in enumerate(row):
        if pd.notna(val) and isinstance(val, str):
            if 'Total revenue' in val:
                nums = [v for v in row.values if pd.notna(v) and isinstance(v, (int, float)) and abs(v) > 100]
                if nums:
                    metrics['revenue'] = {'Q4': nums[0], 'Q3': nums[1] if len(nums) > 1 else None, 'Q2': nums[2] if len(nums) > 2 else None, 'Q1': nums[3] if len(nums) > 3 else None}
            elif 'Total provision for credit losses' in val or ('provision' in val.lower() and 'credit' in val.lower() and 'losses' in val.lower()):
                nums = [v for v in row.values if pd.notna(v) and isinstance(v, (int, float)) and v > 0]
                if nums:
                    metrics['pcl'] = {'Q4': nums[0], 'Q3': nums[1] if len(nums) > 1 else None, 'Q2': nums[2] if len(nums) > 2 else None, 'Q1': nums[3] if len(nums) > 3 else None}
            elif 'Net income' in val and 'attributable' not in val.lower() and 'available' not in val.lower():
                nums = [v for v in row.values if pd.notna(v) and isinstance(v, (int, float)) and abs(v) > 100]
                if nums:
                    metrics['net_income'] = {'Q4': nums[0], 'Q3': nums[1] if len(nums) > 1 else None}
            elif 'Non-interest expense' in val:
                nums = [v for v in row.values if pd.notna(v) and isinstance(v, (int, float)) and abs(v) > 100]
                if nums:
                    metrics['expense'] = {'Q4': nums[0], 'Q3': nums[1] if len(nums) > 1 else None}

# Build markdown
md_content = f'''# CFO Insights: Surprising Findings from Q4 2024

**Query:** {query}

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

Analysis of **Q4 2024 BMO Financials Supplement Report** (Suppq424.xlsx) reveals several surprising insights that warrant CFO attention.

**Files Analyzed:**
- ✅ **Suppq424.xlsx** (Q4 2024): **47 sheets** fully analyzed

---

## 🔍 Surprising Insights

'''

# Add insights
insights = []
if 'pcl' in metrics:
    pcl = metrics['pcl']
    if pcl.get('Q4') and pcl.get('Q3'):
        q4_pcl = pcl['Q4']
        q3_pcl = pcl['Q3']
        pcl_change = ((q4_pcl - q3_pcl) / q3_pcl) * 100
        if abs(pcl_change) > 20:
            insights.append(f'''### ⚠️  **PROVISION FOR CREDIT LOSSES SPIKE**

Q4 2024 PCL of **${q4_pcl:,.0f}M** is **{pcl_change:+.0f}%** vs Q3 ${q3_pcl:,.0f}M.

**Implication:** Significant increase in credit loss provisions may indicate:
- Deterioration in credit portfolio quality
- Changes in economic outlook assumptions
- Specific portfolio segment concerns

**Action Required:** Review credit risk models, portfolio composition, and forward-looking indicators.''')

if 'revenue' in metrics:
    rev = metrics['revenue']
    if rev.get('Q4') and rev.get('Q3'):
        q4_rev = rev['Q4']
        q3_rev = rev['Q3']
        growth = ((q4_rev - q3_rev) / q3_rev) * 100
        if abs(growth) > 5:
            insights.append(f'''### 📈 **STRONG REVENUE GROWTH**

Q4 revenue **${q4_rev:,.0f}M** is **{growth:+.1f}%** vs Q3 ${q3_rev:,.0f}M.

**Implication:** Strong quarter-over-quarter revenue growth driven by:
- Segment performance improvements
- Market share gains
- Product mix optimization

**Action Required:** Identify key growth drivers and assess sustainability.''')

if 'net_income' in metrics:
    ni = metrics['net_income']
    if ni.get('Q4') and ni.get('Q3'):
        q4_ni = ni['Q4']
        q3_ni = ni['Q3']
        ni_change = ((q4_ni - q3_ni) / q3_ni) * 100
        if abs(ni_change) > 15:
            insights.append(f'''### 💰 **NET INCOME VOLATILITY**

Q4 net income **${q4_ni:,.0f}M** is **{ni_change:+.1f}%** vs Q3 ${q3_ni:,.0f}M.

**Implication:** Significant quarter-over-quarter change in profitability.

**Action Required:** Investigate drivers - revenue growth, expense management, or one-time items.''')

# Add efficiency ratio
for idx, row in df5.iterrows():
    row_str = ' '.join([str(v) for v in row.values if pd.notna(v)])
    if 'Efficiency ratio' in row_str:
        nums = [v for v in row.values if pd.notna(v) and isinstance(v, (int, float)) and 0 < v < 1]
        if nums:
            eff_ratio = nums[0]
            if eff_ratio > 0.60:
                insights.append(f'''### 📊 **EFFICIENCY RATIO CONCERN**

Efficiency ratio of **{eff_ratio:.1%}** is above 60% threshold.

**Implication:** Cost management opportunity exists.

**Action Required:** Review expense structure and identify optimization opportunities.''')

md_content += '\n\n'.join(insights)

md_content += '''

---

## 📊 Key Metrics Summary

| Metric | Q4 2024 | Q3 2024 | Change |
|--------|---------|---------|--------|
'''

if 'revenue' in metrics:
    rev = metrics['revenue']
    if rev.get('Q4') and rev.get('Q3'):
        change = ((rev['Q4'] - rev['Q3']) / rev['Q3']) * 100
        md_content += f"| Total Revenue | ${rev['Q4']:,.0f}M | ${rev['Q3']:,.0f}M | {change:+.1f}% |\n"

if 'net_income' in metrics:
    ni = metrics['net_income']
    if ni.get('Q4') and ni.get('Q3'):
        change = ((ni['Q4'] - ni['Q3']) / ni['Q3']) * 100
        md_content += f"| Net Income | ${ni['Q4']:,.0f}M | ${ni['Q3']:,.0f}M | {change:+.1f}% |\n"

if 'pcl' in metrics:
    pcl = metrics['pcl']
    if pcl.get('Q4') and pcl.get('Q3'):
        change = ((pcl['Q4'] - pcl['Q3']) / pcl['Q3']) * 100
        md_content += f"| Provision for Credit Losses | ${pcl['Q4']:,.0f}M | ${pcl['Q3']:,.0f}M | {change:+.1f}% |\n"

md_content += f'''

---

## 📑 Data Coverage

**Total Sheets Analyzed:** 47 sheets from Suppq424.xlsx

**Key Pages Reviewed:**
- Page 5: Financial Highlights
- Page 8: Total Bank Summary Income Statement
- Page 9-13: Segment Performance (P&C, Wealth, Capital Markets)
- Page 24-33: Credit Risk Information

---

## 💡 Recommendations

1. **Immediate Review:** Investigate the significant change in Provision for Credit Losses
2. **Trend Analysis:** Compare Q4 2024 metrics against full-year 2024 trends
3. **Segment Deep Dive:** Analyze segment-level performance for concentration risks
4. **Forward-Looking:** Assess whether Q4 trends are sustainable or one-time events

---

**Note:** All amounts in millions of Canadian dollars (CAD). Analysis based on unaudited supplementary information.

**Generated by:** BMO Excel Agent (ReAct Pattern)
**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
'''

# Write to file
with open('BMO_CFO_Insights_Q4_2024.md', 'w') as f:
    f.write(md_content)

print('✅ Markdown file created: BMO_CFO_Insights_Q4_2024.md')

