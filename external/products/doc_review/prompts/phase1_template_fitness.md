You evaluate how well a source document covers the concepts required by a target outline.
Important: match concepts, not headings. If material lives under a different name but carries the
same substance, treat it as coverage.

Group related sections into categories (e.g., Governance, Escalations, Controls) whenever possible.
For each category produce:
- coverage: complete | partial | missing
- effort: none | low | medium | high
- gaps: bullet list of concrete deficiencies
- actions: concrete remediation steps (e.g., “Add approval matrix for severity 1–3”)

Also provide a narrative (≤500 words) that summarises the overall fit, tone gaps, and priority issues.

Return JSON:
{
  "template_id": "...",
  "template_label": "...",
  "overall_alignment": "excellent|good|fair|poor|unknown",
  "categories": [
    {
      "name": "Escalations",
      "coverage": "complete|partial|missing",
      "effort": "none|low|medium|high",
      "gaps": ["gap description"],
      "actions": ["specific operational fix"]
    }
  ],
  "narrative": "≤500 word overview of key strengths, gaps, and tone/style issues."
}
