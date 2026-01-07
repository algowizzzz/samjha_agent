# TLDR: SQL & Workflow Builder Features

**Target Audience:** Business Product Manager  
**Last Updated:** January 2026

---

## Executive Summary

The platform provides two core capabilities:

1. **SQL Agent** - Converts natural language questions into SQL queries for structured data analysis
2. **Workflow Builder** - Processes documents through AI-powered chains and exports results to markdown

Both features enable non-technical users to extract insights from data and documents without writing code.

---

## Part 1: SQL Agent Features

### What It Does

The SQL Agent allows users to ask business questions in plain English and automatically generates SQL queries to answer them.

**Example:**
- **User asks:** "Show me total sales by country for Q4 2024"
- **System generates:** SQL query that aggregates sales data
- **User receives:** Formatted table with results

### Key Features

#### 1. Natural Language to SQL Conversion
- **Input:** Plain English questions
- **Process:** AI analyzes query → builds structured query spec → generates SQL
- **Output:** Executed SQL results in table format

#### 2. Query Specification System
The system uses a structured "Query Spec" contract that includes:
- **Business Question:** What the user wants to know
- **Dimensions:** Grouping fields (e.g., country, product category)
- **Metrics:** Calculated values (e.g., total sales, average price)
- **Filters:** Conditions (e.g., date ranges, specific values)
- **Time Range:** Temporal constraints
- **Start Table:** Base data source

#### 3. Intelligent Query Planning
- **Gap Detection:** Identifies missing information needed to answer the query
- **Auto-Investigation:** Runs tools to discover schema, preview data, search glossaries
- **Validation:** Ensures query is safe and complete before execution

#### 4. Multi-Step Query Resolution
- **Clarification:** Asks users when information is ambiguous
- **Retry Logic:** Automatically fixes errors and retries
- **Follow-up Support:** Remembers context from previous queries

### How It Works

```
User Query
    ↓
[Decider] - Analyzes query, builds Query Spec
    ↓
[Executor] - Investigates data, generates SQL
    ↓
[Validation] - Safety checks, policy enforcement
    ↓
[Execution] - Runs SQL on DuckDB/Parquet data
    ↓
[Results] - Returns formatted table
```

### Supported Data Sources
- Parquet files
- CSV files
- DuckDB databases
- Views created from data warehouse

### Use Cases
- Business intelligence queries
- Data exploration
- Report generation
- Ad-hoc analysis

---

## Part 2: Workflow Builder Features

### What It Does

The Workflow Builder processes documents (PDFs, Word, etc.) through customizable AI-powered chains and exports results to markdown files.

**Example:**
- **User uploads:** 100 contract PDFs
- **Workflow processes:** Extracts obligations, clauses, dates
- **User receives:** Markdown files with structured analysis

### Key Features

#### 1. Document Ingestion
- **Supported Formats:** PDF, DOCX, Markdown
- **Vision Support:** Can process scanned documents/images
- **Batch Processing:** Handles multiple documents in parallel

#### 2. Chain-Based Processing
- **Chains:** Reusable sequences of AI processing steps
- **Steps:** Each step performs a specific task (extract, analyze, transform)
- **Configurable:** Custom prompts, model selection, temperature settings

#### 3. Workflow Management
- **Workflows:** Combine ingestion profile + chain + export profile
- **Versioning:** Track workflow changes over time
- **Domain Association:** Organize workflows by business domain

#### 4. Export to Markdown
- **Format:** Clean, structured markdown files
- **Multi-Document:** Creates separate .md files per document
- **ZIP Support:** Bundles multiple markdown files when needed
- **Structure:** Organized by document and processing step

### How It Works

```
Document Upload
    ↓
[Ingestion] - Converts to markdown (R0.md)
    ↓
[Chain Execution] - Runs AI steps sequentially
    Step 1: Extract content → R1.md
    Step 2: Analyze structure → R2.md
    Step 3: Generate report → R3.md
    ↓
[Export] - Formats final results
    ↓
[Download] - Markdown file(s) ready
```

### Workflow Components

#### Ingestion Profile
- Defines how documents are ingested
- Specifies accepted file types
- Configures vision processing (if needed)

#### Chain
- Sequence of processing steps
- Each step has:
  - **Title:** Step name
  - **Prompt:** AI instructions
  - **Model Config:** Which AI model to use
  - **Required Inputs:** Which previous step outputs to use

#### Export Profile
- Defines output format (Markdown, JSON, CSV, Excel, DOCX, PDF)
- For markdown: Creates structured .md files
- Organizes results by document and step

### Use Cases
- Contract analysis
- Document review and summarization
- Bulk document processing
- Compliance checking
- Knowledge extraction

---

## Part 3: Markdown Integration

### How Markdown is Used

#### In SQL Agent
- **Domain Files:** Business context stored as markdown
- **Documentation:** Query results can be formatted as markdown
- **Reports:** Export query results to markdown format

#### In Workflow Builder
- **Intermediate Format:** Documents converted to markdown (R0.md)
- **Step Outputs:** Each processing step produces markdown (R1.md, R2.md, etc.)
- **Final Export:** Results exported as markdown files
- **Structure:** Organized by document ID and step index

### Markdown File Structure

**Example Workflow Output:**
```markdown
# Document: doc_abc123

## Step 1
[Content extracted from document]

## Step 2
[Analysis results]

## Step 3
[Final report]
```

### Export Formats Available
1. **Markdown (MD)** - Human-readable, structured text
2. **JSON** - Machine-readable, structured data
3. **CSV** - Tabular data extraction
4. **Excel (XLSX)** - Multi-sheet spreadsheets
5. **DOCX** - Formatted Word documents
6. **PDF** - Print-ready documents

---

## Technical Architecture

### SQL Agent Architecture
- **Decider/Executor Pattern:** Separates reasoning from execution
- **Controller Loop:** Manages retries and state
- **Tool Registry:** Modular tool system
- **Schema Validation:** Ensures query spec completeness

### Workflow Builder Architecture
- **Service Layer:** WorkflowService, IngestionService, ExportService
- **Database:** PostgreSQL for workflow metadata
- **File Storage:** Local filesystem for documents and results
- **Worker System:** Background processing for long-running workflows

---

## Key Benefits

### For Business Users
- **No Coding Required:** Natural language queries and visual workflow builder
- **Fast Results:** Automated processing vs. manual analysis
- **Scalable:** Process hundreds of documents or run complex queries
- **Consistent:** Standardized workflows ensure quality

### For Technical Teams
- **Extensible:** Add custom tools and chains
- **Versioned:** Track changes to workflows
- **Observable:** Logging and error tracking
- **Modular:** Reusable components (chains, profiles)

---

## Limitations & Considerations

### SQL Agent
- Requires structured data (Parquet/CSV)
- Limited to supported data sources
- Complex queries may need clarification
- Performance depends on data size

### Workflow Builder
- Chain creation currently requires programmatic access (Python)
- Workflow creation available via API
- Processing time scales with document count
- AI model costs accumulate with usage

---

## Getting Started

### SQL Agent
1. Ensure data is in Parquet/CSV format
2. Create or select an agent with domain configuration
3. Ask questions in natural language
4. Review and refine queries as needed

### Workflow Builder
1. Create ingestion profile (defines input format)
2. Create chain (defines processing steps) - *currently programmatic*
3. Create export profile (defines output format)
4. Create workflow (combines all three)
5. Upload documents and run workflow
6. Download markdown results

---

## Summary

**SQL Agent:** Natural language → SQL → Data insights  
**Workflow Builder:** Documents → AI Processing → Markdown exports

Both features enable non-technical users to extract value from data and documents through AI-powered automation, with markdown serving as a key output format for human-readable results.

