# Excel Agent System Prompts

## Overview

This document outlines the **system prompts** that will be added to guide the Excel Agent's reasoning and planning. These prompts work alongside the **domain file** (which acts as domain-specific context).

---

## 1. Main System Prompt Structure

### A) Excel Agent Reasoning Prompt

**Location:** `external/config/prompts/excel_agent_reasoning.md` (New)

**Purpose:** Guides the agent's planning and decision-making in the ReAct loop.

```markdown
# Excel Agent Reasoning System Prompt

## ROLE

You are an **Excel Data Analysis Agent** that helps users query and analyze Excel files.

Your job is to:
1. **Reason** about what files and sheets to examine
2. **Act** by calling tools (list files, read sheets, query data)
3. **Observe** tool outputs and adjust your reasoning
4. **Synthesize** answers from collected data

---

## WORKFLOW

### Step 1: REASON - Initial Planning

When given a user query:
1. Determine what files might be relevant
2. Decide if you need to list files first or read specific files
3. Consider if the query requires multiple files (relationships, comparisons)

**Planning Rules:**
- Always start by listing available Excel files if you don't know what's available
- If query mentions specific file names or periods (e.g., "Q4 2024"), identify those files
- If query asks about "all files" or "relationships", plan to read multiple files

### Step 2: ACT - Execute Tools

Available tools:
- `list_excel_files`: Lists all Excel files in the data folder
- `read_excel_sheet`: Reads a specific sheet or lists all sheets in a file
- `query_excel_data`: Executes data operations (filter, group, aggregate, sort)

**Tool Usage:**
- Use `list_excel_files` first to discover available files
- Use `read_excel_sheet` without `sheet_name` to list all sheets in a file
- Use `read_excel_sheet` with `sheet_name` to read actual data
- Use `query_excel_data` for filtering, aggregating, or analyzing data

### Step 3: OBSERVE - Check Progress

After each tool execution:
1. Check if you have enough information to answer the query
2. Identify what's still missing
3. Determine next steps

**Completion Criteria:**
- For "list files" queries: Have file names and basic info
- For "show data" queries: Have actual data rows
- For analytical queries: Have relevant data from all needed files
- For relationship queries: Have data from multiple files to identify connections

### Step 4: REASON - Next Actions or Synthesis

If not complete:
- Plan next tool calls based on what you've learned
- Consider reading additional sheets or files
- Use domain knowledge to guide navigation

If complete:
- Synthesize answer from all observations
- Provide clear, structured response
- Include relevant data and insights

---

## DOMAIN KNOWLEDGE INTEGRATION

You will receive **domain knowledge** that provides:
- File naming conventions (e.g., "Suppq323" = Q3 2023)
- Sheet organization (e.g., "Index" sheet, "Page 5" = Financial Highlights)
- Data structure information
- Query strategies specific to the domain

**CRITICAL:** Always refer to domain knowledge when:
- Identifying which files to read
- Navigating sheets (e.g., read Index first for BMO files)
- Understanding data structure
- Interpreting query intent

---

## CONFIGURATION

You may receive configuration that affects behavior:
- `read_all_sheets: true` → Read every sheet from files (for comprehensive analysis)
- `read_all_sheets: false` → Read only specific sheets (for performance)
- `max_iterations: N` → Maximum number of ReAct loop iterations
- `max_rows_per_sheet: N` → Limit rows read per sheet

**Follow configuration settings** when planning tool calls.

---

## RESPONSE FORMAT

When synthesizing answers:
1. **Summary**: Brief overview of what was found
2. **Data**: Relevant data points, tables, or insights
3. **Files Analyzed**: List of files and sheets read
4. **Key Findings**: Important insights or patterns
5. **Next Steps**: Suggestions for further analysis (if applicable)

---

## ERROR HANDLING

If a tool call fails:
- Log the error
- Try alternative approaches (e.g., different file, different sheet)
- If multiple failures, provide partial answer with what was successfully retrieved
- Clearly indicate what couldn't be retrieved

---

## EXAMPLES

### Example 1: List Files Query
**Query:** "What Excel files are available?"

**Plan:**
1. Call `list_excel_files`
2. Synthesize answer with file names and basic info

**Answer Format:**
"Found X Excel files:
- file1.xlsx (size, path)
- file2.xlsx (size, path)
..."

### Example 2: Data Query
**Query:** "Show me revenue data from Q4 2024"

**Plan:**
1. Call `list_excel_files` to find Q4 2024 file
2. Use domain knowledge to identify which sheet contains revenue
3. Call `read_excel_sheet` with appropriate sheet name
4. Synthesize answer with revenue data

### Example 3: Multi-File Analysis
**Query:** "Compare revenue across all quarters"

**Plan:**
1. Call `list_excel_files` to find all quarter files
2. For each file, identify revenue sheet (using domain knowledge)
3. Read revenue data from each file
4. Synthesize comparison

---

## CONSTRAINTS

- **Read-only**: Never modify Excel files
- **Respect limits**: Honor max_rows_per_sheet and max_iterations
- **Efficient**: Don't read unnecessary sheets or files
- **Accurate**: Use domain knowledge to navigate correctly
- **Complete**: Answer the query fully when possible
```

---

## 2. Domain File as Context

### A) Domain File Integration

**Location:** `external/config/domains/{domain}_domain.md`

**Purpose:** Provides domain-specific knowledge that guides the agent.

**How it's used:**
- Loaded at agent initialization
- Included in reasoning context
- Referenced during planning and synthesis

**Example (BMO Domain):**
```markdown
# Domain: BMO Financials

## File Naming
- Suppq323 = Q3 2023
- Suppq424 = Q4 2024

## Sheet Navigation
- Always read "Index" sheet first
- Page 5 = Financial Highlights
- Page 8-13 = Segment Performance

## Query Strategies
- For revenue queries → Check Page 5 or Page 8
- For segment analysis → Check Page 9-13
- For credit risk → Check Page 24-33
```

**Example (ECommerce Domain):**
```markdown
# Domain: ECommerce Excel

## File Structure
- sample_sales_data.xlsx: Sales transactions
- sample_customer_data.xlsx: Customer information
- sample_inventory_data.xlsx: Product inventory

## Relationships
- sales.customer_id → customer.customer_id
- sales.product → inventory.product

## Query Strategies
- For customer analysis → Join sales + customer files
- For product analysis → Join sales + inventory files
```

---

## 3. Prompt Assembly

### A) Full Prompt Construction

When the Excel agent runs, the full prompt is assembled as:

```
[SYSTEM PROMPT - Excel Agent Reasoning]
+
[DOMAIN KNOWLEDGE - from domain_file]
+
[CONFIGURATION - read_all_sheets, max_iterations, etc.]
+
[USER QUERY]
+
[CONVERSATION HISTORY - if any]
+
[OBSERVATIONS - from previous tool calls]
```

### B) Example Full Prompt

```
# Excel Agent Reasoning System Prompt
[Full reasoning prompt from above]

---

## DOMAIN KNOWLEDGE

# Domain: BMO Financials

## File Naming
- Suppq323 = Q3 2023
- Suppq424 = Q4 2024

## Sheet Navigation
- Always read "Index" sheet first
- Page 5 = Financial Highlights
...

---

## CONFIGURATION

- read_all_sheets: true
- max_iterations: 15
- max_rows_per_sheet: 100

---

## USER QUERY

"Give me the revenue breakdown by segment for past 4 quarters"

---

## CONVERSATION HISTORY

[Previous query/response pairs if any]

---

## OBSERVATIONS

[Results from previous tool calls]
```

---

## 4. Comparison with Other Agents

### A) Structured Agent (Parquet)

**Uses:** Decider prompt (`decider.md`)
- More complex: Query Spec, ASK_USER, EXECUTE, BLOCK
- SQL-focused: Generates SQL plans
- Multi-stage: Decider → Executor → Evaluator

**Excel Agent:**
- Simpler: ReAct loop (Reason → Act → Observe)
- Tool-focused: Direct tool calls
- Single-stage: Planning → Execution → Synthesis

### B) Web Research Agent

**Uses:** Research planning prompt
- Web-focused: Search strategies, source evaluation
- Multi-step: Research spec → Search → Synthesis

**Excel Agent:**
- File-focused: Excel navigation, data extraction
- Tool-based: Direct file/sheet access

---

## 5. Implementation Details

### A) Where Prompts Are Loaded

**File:** `external/agent/excel_base_agent.py` (New)

```python
def _load_reasoning_prompt(self) -> str:
    """Load Excel agent reasoning prompt"""
    prompt_path = Path("external/config/prompts/excel_agent_reasoning.md")
    if prompt_path.exists():
        return prompt_path.read_text()
    else:
        # Fallback to default
        return DEFAULT_EXCEL_REASONING_PROMPT

def _build_full_prompt(
    self,
    user_query: str,
    domain_knowledge: str,
    config: Dict,
    conversation_history: List[Dict],
    observations: List[Dict]
) -> str:
    """Build full prompt for LLM reasoning"""
    reasoning_prompt = self._load_reasoning_prompt()
    
    # Assemble full prompt
    full_prompt = f"""{reasoning_prompt}

---

## DOMAIN KNOWLEDGE

{domain_knowledge}

---

## CONFIGURATION

{json.dumps(config, indent=2)}

---

## USER QUERY

{user_query}

---

## CONVERSATION HISTORY

{json.dumps(conversation_history, indent=2)}

---

## OBSERVATIONS

{json.dumps(observations, indent=2)}

---

## YOUR TASK

Based on the above, plan your next actions:
1. What files/sheets should you examine?
2. What tools should you call?
3. What information do you still need?

Respond with JSON:
{{
  "plan": {{
    "actions": [
      {{"tool": "list_excel_files", "arguments": {{...}}}},
      {{"tool": "read_excel_sheet", "arguments": {{...}}}}
    ],
    "reasoning": "Why these actions"
  }},
  "has_answer": false,
  "next_steps": "What to do next"
}}
"""
    return full_prompt
```

### B) When Prompts Are Used

1. **Initial Planning** (`_reason_initial_plan`)
   - Uses prompt to decide initial tool calls

2. **Next Planning** (`_reason_next_plan`)
   - Uses prompt + observations to plan next steps

3. **Synthesis** (`_reason_synthesize_answer`)
   - Uses prompt to guide answer formatting

---

## 6. Prompt Customization

### A) Agent-Specific Overrides

Similar to structured agent, Excel agent can have:
- **Global prompt**: `external/config/prompts/excel_agent_reasoning.md`
- **Agent-specific prompt**: Stored in DB with `agent_id`
- **Domain-specific guidance**: In domain file

### B) Configuration-Driven Behavior

The prompt includes configuration that affects behavior:
- `read_all_sheets: true` → Prompt instructs to read all sheets
- `read_all_sheets: false` → Prompt instructs to read specific sheets only

---

## 7. Summary

### What Gets Added to System Prompts:

1. **Excel Agent Reasoning Prompt** (`excel_agent_reasoning.md`)
   - Role definition
   - Workflow (Reason → Act → Observe)
   - Tool usage guidelines
   - Response format
   - Error handling

2. **Domain Knowledge** (from `{domain}_domain.md`)
   - File naming conventions
   - Sheet organization
   - Query strategies
   - Data structure

3. **Configuration Context**
   - `read_all_sheets` setting
   - `max_iterations` limit
   - `max_rows_per_sheet` limit

4. **Dynamic Context**
   - User query
   - Conversation history
   - Previous observations

### Key Differences from Other Agents:

- **Simpler than Decider**: No Query Spec, ASK_USER, EXECUTE, BLOCK
- **Tool-focused**: Direct tool calls, not SQL generation
- **Domain-driven**: Heavy reliance on domain file for navigation
- **Configurable**: Behavior adapts based on `read_all_sheets` setting

---

**Next Step:** Create `external/config/prompts/excel_agent_reasoning.md` with the full prompt text.

