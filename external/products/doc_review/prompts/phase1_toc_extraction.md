You are extracting a table of contents from the first few pages of a document.
Use only the provided excerpt (roughly the first 5 pages). If page numbers are
not explicitly stated you may leave them null. Preserve the order in which items
appear and infer heading level (1-5) based on formatting cues such as numbering,
indentation, or heading markers.

Return JSON with:
{
  "entries": [
    {
      "title": "Section title as written",
      "level": 1-5,
      "page_number": null or integer,
      "notes": "optional description",
      "confidence": "high|medium|low"
    }
  ],
  "source": "first_five_pages",
  "truncated": true|false,
  "warnings": ["optional issues noticed"]
}

