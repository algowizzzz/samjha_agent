You receive a single section excerpt alongside the template expectation for that section.
Evaluate how well the section fulfills the template's intent, highlighting concrete gaps
and operational remediation guidance.

Return JSON:
```
{
  "section_title": "string",
  "fit": "good" | "partial" | "none",
  "severity": "low" | "medium" | "high",
  "issues": [
    {
      "id": "SEC-001",
      "title": "Short label",
      "severity": "low|medium|high",
      "type": "grammar|clarity|structural|missing_content|terminology|tone|compliance_precision",
      "location_instruction": "Page/paragraph guidance (or null)",
      "original_text": "Exact text to replace (leave empty for missing content)",
      "suggested_text": "Replacement or insertion text (≤120 words)",
      "reason": "One sentence referencing evidence from the section"
    }
  ],
  "improvement_guidance": [
    "Actionable instruction phrased as an imperative (max 5 items)"
  ]
}
```

Rules:
- Reference specific phrases from the section when citing gaps.
- Only generate ids `SEC-###` (three digits). Increment per issue.
- If the section is entirely missing, set `fit="none"`, `severity="high"`, and produce a single `missing_content` issue summarising what must be created.
- Keep `suggested_text` prescriptive and professional (policy tone).

