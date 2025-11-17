You are reviewing the document's table of contents and structure.
- Identify whether an explicit TOC exists (and what it is called) or infer one from headings.
- Comment on ordering, hierarchy, and flow.
- Highlight missing or misplaced sections.

Return JSON:
{
  "toc_present": true|false,
  "toc_label": "Exact label or null",
  "structure_score": "excellent|good|fair|poor",
  "entries": [
    {"title": "...", "level": 1-5, "page_number": null or number, "notes": "...", "confidence": "high|medium|low"}
  ],
  "observations": ["what is working well"],
  "gaps": ["specific issues to fix"]
}

