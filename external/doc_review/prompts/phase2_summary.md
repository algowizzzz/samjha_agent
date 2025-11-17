Summarise the per-section review results for the document owner.

Inputs you will receive:
- `section_reviews`: array of objects with `section_title`, `fit`, `severity`, `issues`, `improvement_guidance`
- `total_issues` and `high_severity_count`
- optional document metadata (title, purpose)

Return JSON:
```
{
  "overall_posture": "ready" | "needs_work" | "needs_overhaul",
  "section_heatmap": {
    "Overview": "low|medium|high",
    "Scope": "low|medium|high"
  },
  "systemic_gaps": ["Cross-cutting themes (max 5, most severe first)"],
  "narrative": "≤250 words readable summary referencing concrete sections",
  "total_issues": 12,
  "high_severity_count": 4
}
```

Guidance:
- `overall_posture` should reflect the highest severity observed (e.g., any high-severity gaps in critical sections → `needs_work` or `needs_overhaul`).
- `section_heatmap` must cover every section present in the reviews array.
- `systemic_gaps` should focus on recurring deficiencies (missing owners, unclear approvals, passive voice, etc.).
- Narrative should weave together the top 2–3 risks and acknowledge strengths where applicable.

