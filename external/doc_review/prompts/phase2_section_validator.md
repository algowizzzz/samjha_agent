You receive two candidate excerpts for the same document section:

- `heading_candidate` – text extracted using markdown headings
- `toc_candidate` – text extracted using table-of-contents anchors

Your job is to pick the best option (or merge them) so downstream analysis can rely on a clean section excerpt.

Guidance:
- Prefer the candidate that contains the full section (intro sentence + bullets/subsections) without spilling into the next top-level section.
- If both are incomplete but complementary, merge them (keep original order, remove duplicates).
- If neither candidate contains material relevant to the requested section, mark `is_correct=false`.
- Keep whitespace tidy; trim leading/trailing blank lines.

Return JSON:
```
{
  "is_correct": true|false,
  "chosen_method": "headings" | "toc" | "merged" | "none",
  "boundary_check": "perfect" | "ok" | "incomplete" | "unknown",
  "issues": ["optional warnings"],
  "final_section_text": "clean excerpt (<=1500 words, keep markdown)",
  "page_range": [start_page or null, end_page or null],
  "char_range": [start_char or null, end_char or null],
  "reasoning": "one sentence on why you chose this method"
}
```

If `is_correct=false`, set `chosen_method` to "none" and leave `final_section_text` empty.

