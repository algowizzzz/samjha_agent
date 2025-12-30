# Vision Ingestion Prompt — Document to Markdown

## Goal
You will be given one or more page images of a document.
Convert the content to clean, structured **Markdown**.

## Rules
- Preserve headings and lists.
- Represent tables as Markdown tables.
- Ignore purely decorative elements.
- If a page is mostly blank, write: `> Page appears mostly blank.`

## Output
Return Markdown only.
