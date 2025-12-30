# Step 1 — Extract Obligations (MUST/SHOULD)

## Goal
Given the input document content, extract all explicit requirement statements and classify them.

## Instructions
- Output **valid JSON** only.
- Produce an array `obligations`.
- For each obligation include:
  - `type`: MUST | SHOULD | MAY | OTHER
  - `statement`: the exact requirement text (normalize whitespace; keep meaning)
  - `section`: best-guess section heading
  - `evidence_quote`: short supporting quote (<= 25 words)

## Output Schema
{
  "obligations": [
    {
      "type": "MUST",
      "statement": "...",
      "section": "...",
      "evidence_quote": "..."
    }
  ]
}
