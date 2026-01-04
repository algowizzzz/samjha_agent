# CSV Task Processing Chain (2 Steps, Markdown Output)

## Workflow Template Details
- **Name:** Task Analysis & Recommendations
- **Description:** Process task tickets from CSV, analyze issues, and generate recommendations
- **Accepted Document Types:** CSV
- **Output Format:** Markdown (MD)
- **Number of Steps:** 2

---

## Step 1: Task Analysis
**Title:** Analyze Task

**Prompt:**
```
You are analyzing a support ticket. Review the CSV row data provided in the R0 Input section below.

Extract and analyze the following information from the ticket:
- Ticket ID
- Customer name
- Issue description
- Priority level

Based on this information, provide:
1. **Category**: What type of task is this? (e.g., Document Processing, Data Extraction, Format Conversion, Compliance Review)
2. **Key Requirements**: List 2-3 main requirements or goals from the issue description
3. **Complexity Assessment**: Rate as Low, Medium, or High and explain why
4. **Dependencies**: What other information or resources might be needed to complete this task?

Format your response in clear markdown with headers and bullet points.
```

**Model:** claude-haiku-4-5-20251001
**Temperature:** 0.3
**Max Tokens:** 1000
**Required Inputs:** R0 (CSV row data)

---

## Step 2: Generate Recommendations
**Title:** Generate Recommendations

**Prompt:**
```
Based on the task analysis provided in R1, generate actionable recommendations for handling this ticket.

Review the original ticket data from R0 Input and the analysis from R1 Input.

**Provide:**
1. **Recommended Approach**: Step-by-step approach to resolve the issue (numbered list)
2. **Estimated Effort**: Time/resources needed (Low/Medium/High) with justification
3. **Risk Factors**: Potential challenges or blockers that might arise
4. **Next Steps**: Specific, actionable next steps to take (numbered list)

Format as a structured markdown document with clear sections and headers.
```

**Model:** claude-haiku-4-5-20251001
**Temperature:** 0.3
**Max Tokens:** 1200
**Required Inputs:** R0 (CSV row data), R1 (Step 1 output)

---

## How to Create This Workflow

1. Go to Admin Panel → AI Workflow Builder
2. Click "Create New Template"
3. Fill in:
   - **Template Name:** Task Analysis & Recommendations
   - **Description:** Process task tickets from CSV, analyze issues, and generate recommendations
   - **Domains:** Select relevant domains (e.g., Operations, Support)
   - **Accepted Document Types:** Check CSV
   - **Output Formats:** Check Markdown
4. Add Step 1 using the prompt above
5. Add Step 2 using the prompt above
6. Save the template

## Testing
- Use `test_files/sample_tasks.csv` for testing
- Each row will be processed independently
- Output will be markdown recommendations per task

