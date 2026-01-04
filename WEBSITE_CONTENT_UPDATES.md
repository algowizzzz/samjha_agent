# Website Content Updates - Professional Copywriting

This document contains all the content updates for making the platform user-friendly and explaining AI capabilities for bank document and application analysis.

## 1. Home Page (home.html)

### Current vs Updated Content

**MCP Dashboard Box:**
- **Current:** "Manage and monitor MCP tools, view metrics, and access tool configurations."
- **Updated:** "Monitor system health, view tool performance metrics, and access diagnostic information. Essential for system administrators to ensure all AI-powered tools are operating correctly and efficiently."

**Chat Agents Box:**
- **Current:** "Interact with AI agents for data queries, analysis, and natural language interactions."
- **Updated:** "Get instant answers from AI-powered assistants. Ask questions in plain English about your data, documents, or processes. Perfect for quick insights, data exploration, and getting immediate answers to complex questions without writing queries."

**AI Agents at Scale Box:**
- **Current:** "Upload PDFs, run deterministic multi-step prompt chains, and download auditable Markdown outputs."
- **Updated:** "Automate document analysis at scale. Upload multiple documents (policies, reports, applications) and let AI extract key information, summarize content, identify risks, and generate audit-ready outputs. Ideal for compliance reviews, risk assessments, and regulatory reporting. Each analysis creates a complete audit trail with timestamps and version control."

**Admin Panel Box:**
- **Current:** "Configure agent prompts, manage settings, and customize system behavior."
- **Updated:** "Create and manage AI workflow templates. Design custom analysis chains that combine multiple AI steps (summarization, data extraction, risk analysis) to match your specific needs. Configure prompts, set approval workflows, and manage access controls for your team."

## 2. Login Page (login.html)

### Updates Needed:
- Add welcome message
- Better instructions

## 3. AI Bulk Doc Analysis Page (bulk_doc_analysis.html)

### Header Section:
- **Current:** "🤖 AI Assisted Workflow"
- **Updated:** "🤖 AI-Assisted Document Analysis"

### Step 1 Description:
- **Current:** "Click on a workflow to select it"
- **Updated:** "Choose a pre-configured analysis template. Each template combines multiple AI analysis steps designed for specific use cases like regulatory compliance, risk assessment, or policy review."

### Step 2 Description:
- **Current:** Basic upload area
- **Updated:** Add helper text explaining what documents can be analyzed

### Step 3 Description:
- **Current:** "Click on a format to select it"
- **Updated:** "Select your preferred output format. Markdown is recommended for reports and documentation, while JSON is ideal for integration with other systems."

## 4. Admin Panel - Workflow Template Form (admin.html)

### Field Labels and Help Text:

**Template Name:**
- **Label:** "Template Name *"
- **Help Text:** "Give your workflow template a clear, descriptive name (e.g., 'Capital Adequacy Policy Review', 'Loan Application Risk Analysis'). This name will appear in the workflow selection menu."

**Description:**
- **Label:** "Description *"
- **Help Text:** "Explain what this workflow does and when to use it (20-240 characters). Example: 'Analyzes capital adequacy policy documents to extract key requirements, identify compliance gaps, and generate executive summaries for risk management review.'"

**Domains:**
- **Label:** "Business Domains *"
- **Help Text:** "Select the business areas this workflow applies to (comma-separated). Common domains: Risk, Finance, Compliance, Credit, Operations, Legal, Audit. This helps users find the right template for their needs."

**Accepted Document Types:**
- **Label:** "Accepted Document Types *"
- **Help Text:** "Select the file formats this workflow can process. PDF is recommended for documents with complex formatting, while DOCX and TXT work well for simpler documents."

**Processing Steps:**
- **Label:** "Processing Steps *"
- **Help Text:** "Define the AI analysis sequence. Each step processes the output from the previous step. Common patterns: 1) Summarize → 2) Extract tables → 3) Analyze risks, or 1) Extract key facts → 2) Identify compliance requirements → 3) Generate recommendations."

**Step Title:**
- **Help Text:** "Brief, descriptive name for this step (e.g., 'Extract Financial Metrics', 'Summarize Key Requirements', 'Risk Assessment')."

**Step Prompt:**
- **Help Text:** "Detailed instructions for the AI on what to analyze and how to format the output. Be specific about what information to extract, what format to use, and any requirements. Include examples when helpful."

**Model:**
- **Help Text:** "Choose the AI model based on complexity: Claude Sonnet 4 (best for complex analysis), Claude 3.5 Sonnet (balanced performance), Claude 3 Haiku (faster, good for simple tasks)."

**Temperature:**
- **Help Text:** "Controls AI creativity vs consistency (0.0-1.0). Use 0.1-0.3 for factual extraction and summaries (recommended for audit work). Higher values (0.7+) allow more creative analysis but less consistency."

**Max Tokens:**
- **Help Text:** "Maximum length of AI output (100-8192 tokens). Use 1000-2000 for summaries, 4000+ for detailed analysis. One token ≈ 4 characters."

**Output Format:**
- **Label:** "Output Format *"
- **Help Text:** "Select the format for final outputs. Markdown is human-readable and perfect for reports. JSON enables automated processing and integration with other systems."

## 5. Navigation (base.html)

**Nav Labels:**
- **Home:** "Home" (good as is)
- **MCP Dashboard:** "System Dashboard" (clearer for non-technical users)
- **Chat Agents:** "AI Assistant" (more user-friendly)
- **Tools:** "Tools" (good as is)

## 6. Use Case Examples to Add

Add a help section or tooltip explaining common use cases:

### Use Case 1: Regulatory Compliance Review
"Upload policy documents or regulatory guidance. The AI extracts key requirements, identifies gaps in current practices, and generates compliance checklists with audit evidence."

### Use Case 2: Risk Assessment
"Analyze risk reports, policy documents, or application forms. Extract risk factors, summarize mitigation strategies, and generate risk assessment summaries for management review."

### Use Case 3: Data Extraction (5W Analysis)
"Automatically extract Who, What, When, Where, and Why information from documents. Perfect for populating databases, creating summaries, and ensuring consistent data capture across documents."

### Use Case 4: Audit Evidence Generation
"Create complete audit trails by analyzing documents, extracting relevant information, summarizing findings, and generating timestamped, version-controlled outputs suitable for audit documentation."

### Use Case 5: Governance Documentation
"Review governance documents, extract key governance requirements, identify responsibilities and accountabilities, and generate governance summaries for board presentations."

