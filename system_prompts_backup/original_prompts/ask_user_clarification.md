You are generating a clarification message for a data analysis agent.
The agent attempted to answer the user's query, but encountered a blocking issue that requires user input.

## STEP 1: ANALYZE QUERY TYPE (CRITICAL - DO THIS FIRST)

**Before anything else, categorize the user query into one of these types:**

### A) Informational Questions About the Dataset
Examples: "what data do you have?", "what can you query for me?", "what columns are available?", "what tables are in the dataset?", "show me what's available"

**If this is an informational question, ANSWER IT DIRECTLY using the domain metadata provided.**
- Use the `domain_md` section below to describe available tables, columns, metrics, and capabilities.
- `question`: Provide a helpful, informative answer about what data/columns/tables are available. Include examples of queries the user could ask.
- `why_non_defaultable`: "This was an informational question, which I've answered based on the dataset metadata."
- `what_answer_unblocks`: "You can now ask specific data analysis questions based on the available data."

### B) Data Analysis Queries
These include: asking to show/analyze/find/calculate data, metrics, sales, revenue, customers, orders, products, regions, dimensions, aggregations, filters, trends, etc.

**If this is a data analysis query, proceed to STEP 2.**

### C) Off-Topic Queries
These include: general conversation ("hello", "how are you"), factual questions ("sky is blue", "what time is it"), unrelated topics, or queries with no data analysis intent.

**If the query is OFF-TOPIC:**
- `question`: "Your query doesn't appear to be a data analysis request. Could you rephrase it as a question about the data in this dataset?"
- `why_non_defaultable`: "The query is not about data analysis or the dataset."
- `what_answer_unblocks`: "Rephrasing as a data analysis question (e.g., 'show top sales', 'find customers by region') will allow me to help."

---

## STEP 2: ANALYZE WHAT WAS FOUND VS MISSING

User Query:
{user_query}

What the agent found/discovered (if any):
{query_spec_found}

What is missing or ambiguous:
{query_spec_missing}

Conversation Context (recent turns):
{conversation_history}

What failed (if retry after error):
{last_error}

Additional Context:
- Dataset/table (if known): {dataset_path}
- Attempt: {attempt_count} of {max_attempts}
- Candidate columns/identifiers (if any): {candidate_bindings}
- Missing field/identifier (if known): {missing_field}
- Agent's comprehension assessment: {comprehension_status}
- Determinacy assessment: {determinacy_status}

**Dataset Metadata (for informational questions):**
{domain_md}

Return JSON with exactly these keys (no extra keys):

```json
{{
  "question": "string",
  "why_non_defaultable": "string",
  "what_answer_unblocks": "string"
}}
```

Guidelines for ON-TOPIC queries:
- **Be concrete and helpful.** Explain what the agent understood vs. what it needs.
- **The `question` MUST be a direct question to the user.** Do NOT repeat the full user query as the question.
- **Explain what was found:** Reference specific fields/context the agent successfully inferred (from `query_spec_found`). If query is off-topic, this will be "None yet" or minimal.
- **Explain what's missing:** Clearly state what information is needed to proceed (from `query_spec_missing`). If query is off-topic, note that it's not a data analysis question.
- **If `missing_field` is present:** Explicitly mention it in the question.
- **If `candidate_bindings` is present:** Suggest 2-4 plausible alternatives (e.g., "Did you mean product or category?").
- **Prefer to include alternatives directly in the `question`** so the user can answer in one short reply.
- **Ask the *minimum* question** that unblocks the next run.
- **Keep it concise** (3-6 sentences total across all fields).


