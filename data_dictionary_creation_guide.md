# Data Dictionary Creation Guide
## Questionnaire for Business Users

This guide will help you create a comprehensive data dictionary JSON file that enables our AI agent to understand and query your data effectively.

---

## Table of Contents

1. [Detailed Questionnaire](#detailed-questionnaire)
2. [Simplified Quick-Start Template](#simplified-quick-start-template)
3. [Example Structure Reference](#example-structure-reference)

---

# Detailed Questionnaire

## Section 1: Tables

### Table Name
**Question 1.1:** What is your table name?
- Provide the exact table name as it appears in your database/system
- Example: `limits_data`, `sales_data`, `customer_data`

### Table Description
**Question 1.2:** What does this table contain?
- Write 1-2 sentences describing what this table contains and its primary purpose
- Be specific about the data scope
- **Example:** *"Single canonical table containing market risk limits, exposures, governance metadata and audit fields. Use this single table for daily monitoring, breach tracking and reporting."*

### Business Context
**Question 1.3:** How is this table used in your business?
- Describe who uses this table and for what business decisions
- Explain the business role/purpose of this data
- **Example:** *"Source of truth for front-office and risk-monitoring limits. Contains limit definitions, current exposures, utilization, and lightweight audit/approval fields for breaches and temporary overrides."*

---

## Section 2: Key Columns

For each important column in your table, answer the following:

### Basic Column Information
**Question 2.1:** **Column Name**
- What is the exact column name as it appears in the database?

**Question 2.2:** **Definition**
- What does this column represent?
- What business meaning does it have?
- **Example:** *"Snapshot date (YYYY-MM-DD). Default to MAX(date) for latest data."*

**Question 2.3:** **Data Type**
- What type of data is stored? (DATE, VARCHAR, INTEGER, DECIMAL, etc.)

**Question 2.4:** **Usage**
- How should queries filter/use this column?
- Provide example SQL patterns if applicable
- **Example:** *"WHERE date = (SELECT MAX(date) FROM limits_data) for latest, or date >= MAX(date) - INTERVAL 'N days' for trends"*

### Critical Notes
**Question 2.5:** Are there any common misunderstandings about columns?
- Document any critical warnings or common mistakes users make
- **Example:** *"⚠️ CRITICAL: When users mention 'PV01', 'Gamma', 'Liquidity', etc., they mean limit_group, NOT limit_type. Example: 'PV01 limits' → WHERE limit_group = 'PV01', NOT WHERE limit_type = 'PV01' (limit_type is 'PV01 Delta')"*

---

## Section 3: Business Glossary

### Overview/Domain Context
**Question 3.1:** What is the high-level business domain?
- Example: Market Risk, Sales, Inventory, Customer Management, etc.

**Question 3.2:** What are the key business concepts/metrics?
- List each key concept with a brief definition
- **Example:**
  - PV01: "Interest rate sensitivity (dollar value of 1bp rate change)"
  - Delta: "Price sensitivity (change in value per $1 move in underlying)"
  - Utilization: "Exposure / effective_limit ratio. >= 1.0 = breach"

**Question 3.3:** What is the hierarchy or structure?
- Describe how the data is organized hierarchically
- **Example:** *"Trading Desk → Risk Category (limit_group) → Limit Type (limit_type) → Specific Limit"*

### Business Terminology Mapping

**Question 3.4:** How do users express concepts vs. how they're stored?
- For each important business concept:
  - **User says:** List common phrases/variations users might say
  - **Column:** Which database column maps to this?
  - **Exact value:** What is the exact value in the database?
  - **Critical notes:** Any important warnings?
  - **Example:**
    - User says: ["PV01", "PV01 limits", "interest rate risk", "DV01"]
    - Column: `limit_group`
    - Exact value: "PV01"
    - Critical note: "When user says 'PV01' or 'DV01', use limit_group = 'PV01' (NOT limit_type)"

**Question 3.5:** What threshold/level expressions do users use?
- For each threshold level:
  - **User says:** ["breached", "over limit", "exceeded"]
  - **SQL condition:** "utilization >= 1.0"
- **Examples:**
  - Breached: User says ["breached", "over limit"] → SQL: "utilization >= 1.0"
  - High utilization: User says ["high utilization", "near breach", "above 90%"] → SQL: "utilization >= 0.9"
  - Low utilization: User says ["low utilization", "well below limit"] → SQL: "utilization < 0.7"

**Question 3.6:** What time expressions do users use?
- For each time expression:
  - **User says:** ["latest", "current", "most recent", "today"]
  - **SQL pattern:** "date = (SELECT MAX(date) FROM table_name)"
- **Examples:**
  - Latest: ["latest", "current", "most recent"] → "date = (SELECT MAX(date) FROM limits_data)"
  - Past week: ["past week", "last 7 days"] → "date >= (SELECT MAX(date) - INTERVAL '7 days' FROM limits_data)"
  - Past month: ["past month", "last 30 days"] → "date >= (SELECT MAX(date) - INTERVAL '30 days' FROM limits_data)"

---

## Section 4: Procedural Knowledge

### Data Samples
**Question 4.1:** What are typical date ranges in your data?
- Example: "2024-10-09 to 2024-11-08"

**Question 4.2:** What are typical values for key categorical columns?
- List 5-10 examples for each important column
- **Example:**
  - Typical desks: ["Oil Products NGL Trading", "US Delta One", "Canadian Options"]
  - Typical limit groups: ["PV01", "RR & Gamma", "Asset", "Stress Limits"]

### Minimal Unique Attributes
**Question 4.3:** What columns uniquely identify a record?
- List the bare minimum set of columns needed (without using ID columns)
- **Example:** ["letter_nm", "limit_type", "meas_unit", "limit_group", "date", "aggr_func_cd"]

**Question 4.4:** Provide examples of queries that uniquely identify records
- Give 2-3 examples with corresponding SQL filters
- **Example:**
  - Use case: "Get specific PV01 limit for Canadian Options desk"
  - Query: "show me Canadian Options PV01 limit"
  - SQL filter: "letter_nm = 'Canadian Options' AND limit_group = 'PV01' AND limit_type = 'PV01 Delta'"

### Default Behaviors
**Question 4.5:** What defaults should apply when users don't specify?
- For each default:
  - **Parameter:** e.g., "date", "limit_class"
  - **Default value:** e.g., "MAX(date)"
  - **SQL pattern:** "WHERE date = (SELECT MAX(date) FROM table)"
  - **Apply when:** When should this default be used?
- **Example:**
  - Parameter: "date"
  - Default: "MAX(date) - most recent date"
  - SQL: "WHERE date = (SELECT MAX(date) FROM limits_data)"
  - Apply when: "Always (date is almost always needed)"

### Common Query Patterns
**Question 4.6:** What are the most common query types?
- For each pattern:
  - **Pattern name:** e.g., "View limits by desk"
  - **Description:** What do users want to see?
  - **User query examples:** ["show me limits for Canadian Options", "what are the Canadian Money Markets limits"]
  - **SQL template:** "SELECT * FROM table WHERE column = '{placeholder}' AND date = (SELECT MAX(date) FROM table)"

### Critical Field Mappings
**Question 4.7:** Are there commonly confused field mappings?
- Provide correct vs. incorrect examples
- **Example:**
  - User query: "show me Primary PV01 limits"
  - Correct: "WHERE limit_group = 'PV01' AND limit_class = 'Primary'"
  - Incorrect: "WHERE limit_type = 'PV01'"

### Trend Analysis
**Question 4.8:** Should single-record queries show historical trends?
- **Default period:** e.g., "past 5 days"
- **SQL pattern:** Show how to query trends
- **Example:**
  - Trigger: "Query uniquely identifies a single limit"
  - Default period: "past 5 days"
  - SQL: "SELECT date, exposure_amt, utilization FROM table WHERE {filters} AND date >= MAX(date) - INTERVAL '5 days'"

### Extension/Special Handling
**Question 4.9:** Are there special flags or states to monitor?
- For each:
  - What is it?
  - Why is it important?
  - What queries monitor it?
- **Example:**
  - Extension: "Temporary limit override"
  - Importance: "Limits with extension=1 have been approved for temporary increase"
  - Monitoring query: "WHERE extension = 1 AND end_dt <= MAX(date) + INTERVAL '7 days'"

### Clarification Rules
**Question 4.10:** When should the system ask for clarification?
- **When to clarify:**
  - Ambiguous threshold without numeric hint
  - Partial name that could match multiple records
  - Multiple currencies with aggregation
- **When NOT to clarify:**
  - Date not specified (use default)
  - Aggregation keywords present
  - Query is specific enough
- **Specific clarifying questions:**
  - "By 'high utilization', do you mean above 90%, 95%, or breached (100%)?"
  - "Which desk? (List options)"
  - "Which currency? (USD, CAD, EUR, or all?)"

### Follow-Up Query Patterns
**Question 4.11:** How should follow-up questions be handled?
- **Pronoun mapping:** How to interpret "these", "those", "it", etc.
- **Common patterns:**
  - Aggregation on previous results
  - Filter refinement
  - Adding/removing filters
- **Example:**
  - Previous: "show me US limits" (WHERE region_cd = 'AMERICAS')
  - Follow-up: "how many of these are gross"
  - SQL: "WHERE region_cd = 'AMERICAS' AND aggr_func_cd LIKE '%GROSS%'"

### Aggregation Patterns
**Question 4.12:** What keywords do users use for aggregations?
- **Keyword mapping:**
  - "sum" → SUM(column)
  - "count" / "how many" → COUNT(*)
  - "average" / "avg" → AVG(column)
  - "maximum" / "max" → MAX(column)
- **Column mapping:**
  - "exposure" → exposure_amt
  - "limit" → effective_limit
  - "utilization" → utilization
- **Special handling:**
  - "Cannot aggregate different currencies without conversion"

### Query Reference
**Question 4.13:** Provide 10-20 common queries users ask
- For each query:
  - **ID/Title:** e.g., "Q1: Show me latest data"
  - **User query:** "Show me latest data"
  - **SQL:** "SELECT * FROM table WHERE date = (SELECT MAX(date) FROM table)"
  - **Explanation:** Brief description of what the query does

---

# Simplified Quick-Start Template

## Step 1: Table Basics

- **Table Name:** `___________`
- **Description (1-2 sentences):** `___________`
- **Business Context:** `___________`

## Step 2: List Your Columns

For each important column, fill in:

- **Column Name:** `___________`
- **What it means:** `___________`
- **Data Type:** `___________`
- **How to use in queries:** `___________`

### Critical Notes
Any important warnings about columns: `___________`

## Step 3: Business Terms

### Key Business Concepts
List the main business concepts users need to understand:
- Concept 1: `___________`
- Concept 2: `___________`
- Concept 3: `___________`

### How Users Say Things vs. Database Values
- User says "___________" → Database column = `___________`, value = `___________`
- User says "___________" → Database column = `___________`, value = `___________`
- User says "___________" → Database column = `___________`, value = `___________`

### Threshold Expressions
- "breached" / "over limit" → SQL: `utilization >= 1.0`
- "high utilization" / "near breach" → SQL: `utilization >= 0.9`
- Add more as needed: `___________`

### Time Expressions
- "latest" / "current" → SQL: `date = (SELECT MAX(date) FROM table)`
- "past week" / "last 7 days" → SQL: `date >= (SELECT MAX(date) - INTERVAL '7 days' FROM table)`
- Add more as needed: `___________`

## Step 4: Common Queries

List the top 5-10 questions users ask, with example SQL:

1. **User asks:** `___________`
   - **SQL:** `___________`
   
2. **User asks:** `___________`
   - **SQL:** `___________`
   
3. **User asks:** `___________`
   - **SQL:** `___________`

(Continue for all common queries...)

## Step 5: Default Behaviors

- When date not specified: `Use MAX(date)`
- When limit class not specified: `Use 'Primary'`
- Add more defaults: `___________`

## Step 6: Typical Data Values

- **Date range:** `___________`
- **Typical values for key columns:**
  - Column 1: `["value1", "value2", "value3"]`
  - Column 2: `["value1", "value2", "value3"]`

---

# Example Structure Reference

Here's the JSON structure you'll be creating. Use this as a reference:

```json
{
  "tables": {
    "your_table_name": {
      "description": "Your table description here",
      "business_context": "Your business context here",
      "key_columns": {
        "critical_note": "Any critical warnings",
        "column_name_1": {
          "definition": "What this column means",
          "data_type": "DATE/VARCHAR/INTEGER/etc",
          "usage": "How to use in queries"
        },
        "column_name_2": {
          "definition": "...",
          "data_type": "...",
          "usage": "..."
        }
      }
    }
  },
  "business_glossary": {
    "domain_overview": {
      "definition": "High-level business domain explanation",
      "key_metrics": {
        "Metric1": "Definition",
        "Metric2": "Definition"
      },
      "hierarchy": "How data is organized"
    },
    "business_terminology_mapping": {
      "description": "How users express concepts vs database",
      "concept_name": {
        "user_says": ["phrase1", "phrase2"],
        "column": "database_column_name",
        "exact_value": "exact_value_in_db",
        "critical_note": "Any warnings"
      },
      "threshold_levels": {
        "breached": {
          "user_says": ["breached", "over limit"],
          "sql_condition": "utilization >= 1.0"
        }
      },
      "time_expressions": {
        "latest": {
          "user_says": ["latest", "current"],
          "sql_pattern": "date = (SELECT MAX(date) FROM table)"
        }
      }
    }
  },
  "procedural_knowledge": {
    "summary": "Brief summary of procedural knowledge",
    "data_samples": {
      "typical_dates": "date range",
      "typical_column_values": ["value1", "value2"]
    },
    "default_behaviors": {
      "defaults": [
        {
          "parameter": "date",
          "default_value": "MAX(date)",
          "sql_pattern": "WHERE date = (SELECT MAX(date) FROM table)",
          "apply_when": "Always"
        }
      ]
    },
    "common_query_patterns": {
      "patterns": [
        {
          "pattern_name": "Pattern name",
          "description": "What users want",
          "user_query_examples": ["example query 1", "example query 2"],
          "sql_template": "SELECT * FROM table WHERE column = '{placeholder}'"
        }
      ]
    },
    "critical_field_mapping": {
      "description": "Commonly confused mappings",
      "rule": "Correct mapping rule",
      "examples": [
        {
          "user_query": "example query",
          "correct": "correct SQL",
          "incorrect": "incorrect SQL"
        }
      ]
    },
    "clarification_rules": {
      "when_to_clarify": [
        {
          "scenario": "Ambiguous threshold",
          "clarification_question": "Question to ask user"
        }
      ],
      "when_not_to_clarify": [
        {
          "scenario": "Date not specified",
          "action": "Use default"
        }
      ]
    },
    "query_reference": {
      "common_queries": [
        {
          "id": "Q1",
          "title": "Query title",
          "user_query": "Example user query",
          "sql": "SELECT * FROM table",
          "explanation": "What this query does"
        }
      ]
    }
  }
}
```

---

## Next Steps

1. Fill out the questionnaire above for your data
2. Reference the example JSON structure
3. Create your JSON file following the structure
4. Save it as `data_dictionary_[your_domain].json` in the appropriate location
5. Test with sample queries to ensure everything works correctly

## Need Help?

Refer to the example file: `external/config/data_dictionary/data_dictionary_risk_v2.json` for a complete working example.

---

*Last Updated: 2025-01-27*
