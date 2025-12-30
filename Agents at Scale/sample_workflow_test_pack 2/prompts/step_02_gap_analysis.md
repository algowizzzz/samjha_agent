# Step 2 — Gap Analysis

## Goal
Using the extracted obligations JSON and the original document, identify likely gaps or ambiguities.

## Instructions
- Output **valid JSON** only.
- Produce:
  - `gaps`: array of items with `obligation_statement`, `gap_description`, `severity` (Low/Med/High)
  - `questions`: array of clarifying questions to ask a policy owner (if needed)
- Be conservative: if the document already covers the obligation clearly, do not invent gaps.

## Output Schema
{
  "gaps": [
    {
      "obligation_statement": "...",
      "gap_description": "...",
      "severity": "Medium"
    }
  ],
  "questions": ["..."]
}
