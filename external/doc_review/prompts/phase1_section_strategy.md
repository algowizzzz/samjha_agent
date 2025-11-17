You have three inputs:
1. The document executive summary (what it is, key themes).
2. The table of contents review (structure strengths/weaknesses).
3. The template fitness assessment (category-level gaps and effort).

Synthesize these to advise the author on next steps before deeper processing.

Return JSON:
{
  "verdict": "ready" | "needs_improvement",
  "rationale": "Why you chose that verdict",
  "recommended_section_level": "h1"|"h2"|"h3"|"h4"|"h5",
  "fallback_levels": ["..."],
  "estimated_sections": number or null,
  "next_steps": [
    "Actionable instruction (operational wording, e.g., 'Rename Section 4 to Escalation Matrix and add approval ladder')"
  ]
}

Keep the tone concise, actionable, and focused on “how” to fix issues (not abstract comments).

