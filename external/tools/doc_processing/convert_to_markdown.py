from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    convert_from_path = None
    Image = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from tools.base_mcp_tool import BaseMCPTool

from .utils import (
    generate_md_file_id,
    resolve_path,
    write_text_file,
)


_MARKDOWN_OUTPUT_DIR = Path("external/data/doc_review/markdown")


class ConvertToMarkdownTool(BaseMCPTool):
    """Convert documents to Markdown using vision-based LLM transcription."""

    SUPPORTED_TEXT_TYPES = {"md", "markdown", "txt"}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "convert_to_markdown",
            "description": "Converts source documents into Markdown using vision-based transcription.",
            "inputSchema": self.get_input_schema(),
            "outputSchema": self.get_output_schema(),
        }
        if config:
            tool_config.update(config)
        super().__init__(tool_config)
        
        # Initialize Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set - vision conversion will fail")
        self.client = Anthropic(api_key=api_key) if Anthropic and api_key else None

        # Use direct PDF→JSON conversion (bypass markdown)
        self.use_direct_json = os.getenv("USE_DIRECT_PDF_JSON", "true").lower() == "true"

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "source_path": {"type": "string"},
                "file_type": {"type": "string"},
            },
            "required": ["file_id", "source_path", "file_type"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "md_file_id": {"type": "string"},
                "md_path": {"type": "string"},
                "raw_markdown": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["file_id", "md_file_id", "md_path", "raw_markdown"],
            "additionalProperties": False,
        }

    def _convert_text_like(self, source_path: Path) -> str:
        """Convert plain text files."""
        return source_path.read_text(encoding="utf-8")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _transcribe_page_direct_to_json(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """Transcribe PDF page image directly to BlockEditor JSON blocks (bypass markdown)."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized - check ANTHROPIC_API_KEY")
        
        image_base64 = self._image_to_base64(image)
        
        prompt = """You are a **high-accuracy PDF Vision Transcriber and Structural Layout Engine**.

Your job is to convert this **PDF page image** directly into **structured BlockEditor JSON blocks**, preserving *all* visual, semantic, and structural features with absolute fidelity.

# PRIMARY OBJECTIVE

Extract ALL visible text, formatting, and structure from the PDF image and output as a JSON array of block objects.

# CRITICAL RULES

1. **EXACT CONTENT** - Preserve all text exactly as shown (do not fix grammar, spelling, or OCR errors)
2. **ALL ELEMENTS** - Include headers, footers, logos, footnotes, page numbers, stamps, signatures
3. **INLINE FORMATTING** - Detect bold, italic, underline from font styling
4. **FONT SIZE → HEADING LEVEL** - Largest text = level 1, next = level 2, etc.
5. **NO REWRITING** - Output what you see, not what you think it should be
6. **PRESERVE SPACING** - Maintain line breaks, blank lines, indentation

# BLOCK TYPES

### 1. HEADING
Use for large, bold, standalone text. Detect level from visual font size.

```json
{
  "id": "b1",
  "type": "heading",
  "level": 1,
  "content": "Guideline",
  "formatting": {"bold": true, "size": "large"},
  "bbox": [x1, y1, x2, y2]
}
```

### 2. PARAGRAPH
Regular text blocks. Use inline segments for mixed formatting.

```json
{
  "id": "b2",
  "type": "paragraph",
  "content": [
    {"text": "The ", "bold": false},
    {"text": "Bank Act (BA)", "bold": true},
    {"text": " requires...", "bold": false}
  ],
  "bbox": [...]
}
```

**IMPORTANT:** If paragraph has ANY bold/italic/underline within it, use array format with segments.

### 3. FIELD LABELS
Lines like "**Label:** value" are level-3 headings.

```json
{
  "id": "b3",
  "type": "heading",
  "level": 3,
  "content": "Effective Date: November 2023 / January 2024",
  "formatting": {"bold": true}
}
```

### 4. LIST
```json
{
  "id": "b4",
  "type": "bulleted_list",
  "items": [
    {"content": "First item"},
    {"content": "Second item", "children": [{"content": "Nested"}]}
  ]
}
```

### 5. TABLE
```json
{
  "id": "b5",
  "type": "table",
  "columns": ["Name", "Value"],
  "rows": [["Risk Type", "Market"]]
}
```

### 6. SPECIAL ELEMENTS
- `type: "divider"` for horizontal lines
- `type: "image"` for logos/charts  
- `type: "empty"` for blank lines
- `type: "preformatted"` for fixed-width/aligned text
- `type: "code"` for code blocks
- `type: "blockquote"` for quoted sections

### 7. SMALL TEXT
Footnotes, subscripts, fine print:

```json
{
  "type": "paragraph",
  "content": "See footnote 1",
  "formatting": {"size": "small"}
}
```

# HEADING LEVEL DETECTION RULES

- **Very large text (48pt+)**: level 1 (document title)
- **Large bold (24-36pt)**: level 2 (chapter/section)
- **Medium bold (14-18pt)**: level 3 (subsection)  
- **Bold field labels** ("Date:", "Note:"): level 3
- **Smaller bold (12pt)**: level 4

# INLINE FORMATTING RULES

When you see mixed formatting within a line:
- **MUST use array format** with text segments
- Each segment has: `text`, `bold`, `italic`, `underline`, `code`, `superscript`, `subscript`

Example:
"Risk from **credit exposure** is calculated" →
```json
{
  "content": [
    {"text": "Risk from "},
    {"text": "credit exposure", "bold": true},
    {"text": " is calculated"}
  ]
}
```

# ALIGNMENT DETECTION

If visually centered, right-aligned, or justified:
```json
{"alignment": "center"}  // for titles
{"alignment": "right"}   // for dates/page numbers
```

# OUTPUT FORMAT

Return ONLY a valid JSON object:

```json
{
  "blocks": [
    { ... block 1 ... },
    { ... block 2 ... }
  ],
  "page_metadata": {
    "page_number": """ + str(page_num) + """,
    "has_header": true,
    "has_footer": true
  }
}
```

NO markdown code fences. NO explanations. ONLY JSON.

# THINGS YOU MUST NOT DO

❌ Do NOT fix spelling errors
❌ Do NOT merge paragraphs  
❌ Do NOT rewrite content
❌ Do NOT skip headers/footers
❌ Do NOT skip logos or images
❌ Do NOT normalize formatting
❌ Do NOT add words not in the image

Begin transcription. Output ONLY JSON."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Use Sonnet for better accuracy
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            response_text = response.content[0].text.strip()
            
            # Remove markdown code fences if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Parse JSON
            result = json.loads(response_text)
            blocks = result.get('blocks', [])
            
            self.logger.info(f"Page {page_num}: Direct JSON transcription created {len(blocks)} blocks")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed direct JSON transcription for page {page_num}: {e}")
            raise

    def _transcribe_page_with_vision(self, image: Image.Image, page_num: int) -> str:
        """Transcribe a single page image using Claude vision."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized - check ANTHROPIC_API_KEY")
        
        # Convert image to base64
        image_base64 = self._image_to_base64(image)
        
        # Prompt for precise transcription
        prompt = """Transcribe this document page preserving ALL visual formatting and structure. Start immediately with the content.

CRITICAL RULES:
1. NO intro text - start transcribing right away
2. This may continue from a previous page
3. DO NOT ADD # ## ### symbols for headings
4. DO USE **bold** markdown to represent visual formatting
5. Preserve document structure through formatting + line breaks

HEADING TRANSCRIPTION (large, bold, standalone lines):
- Document title (largest, bold) → **Title Text**
- Section heading (large, bold) → **1. Overview**
- Subsection (medium, bold) → **1.1 Details**
- Always: bold format + standalone line + blank line after
(NOTE: Use **bold**, NOT # symbols)

INLINE FORMATTING (within paragraphs):
- Bold text → **text**
- Italic text → *text*
- Bold + Italic → ***text***
- Underlined/emphasized → *text*
- Highlighted/colored background → ==highlighted text==
- Small print / footnotes → <small>footnote text</small>
- Strikethrough → ~~strikethrough~~

STRUCTURE & SPACING:
- Preserve blank lines between sections
- Headings are standalone lines (not part of paragraphs)
- Keep paragraph breaks
- Maintain document flow
- Centered text → <center>centered content</center>
- Right-aligned → <right>right-aligned content</right>
- Indented paragraphs → Use 2 spaces per indentation level

LISTS:
- Bullet points → - Item text
- Numbered items → 1. Item text
- Sub-bullets → Indent with spaces (2 spaces per level):
  - Level 1 item
    - Level 2 item
      - Level 3 item
- Preserve indentation levels

TABLES:
- Convert ALL tables to markdown format with | pipes
- Include header row with column names
- Add separator row with |---|---|---| 
- Example:
  | Column 1 | Column 2 |
  |----------|----------|
  | Data 1   | Data 2   |

SPECIAL ELEMENTS:
- Callout boxes / bordered sections → > Callout: content here
- Blockquotes → > quoted text
- Code blocks → ```code here```
- Horizontal rules → ---

IMAGES/CHARTS/DIAGRAMS:
- Mark as: ![Image: Brief description]
- Do NOT skip images - always mark their presence
- Note position (top/middle/bottom of content)

COLOR & HIGHLIGHTING:
- Yellow highlight → ==text==
- Other colors → ==text== (use == for any highlighting)

DISTINCTION:
✅ Large heading → **Risk Management Overview** (bold, standalone)
✅ Bold word in paragraph → This is **important** text.
✅ Highlighted → This is ==critical== information.
✅ Small print → <small>See footnote 1</small>
✅ Centered → <center>CONFIDENTIAL</center>
❌ NEVER → # Risk Management Overview (no # symbols)

Begin transcription:"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            
            # Extract text from response
            markdown = response.content[0].text.strip()
            self.logger.info(f"Transcribed page {page_num}: {len(markdown)} characters")
            return markdown
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe page {page_num}: {e}")
            return f"[Error transcribing page {page_num}: {str(e)}]"

    def _generate_stable_block_id(self, page: int, block_num: int, content: str) -> str:
        """Generate stable block ID: p{page}_b{block_num}_{hash}"""
        # Hash first 100 chars of content for uniqueness
        content_hash = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return f"p{page}_b{block_num}_{content_hash}"
    
    def _detect_block_type(self, content: str) -> str:
        """Detect block type from content structure with **bold** headings"""
        stripped = content.strip()
        if not stripped:
            return 'empty'
        
        # Check first line for structure
        first_line = stripped.split('\n')[0].strip()
        
        # Bullet list
        if first_line.startswith('- '):
            return 'bullet'
        
        # Numbered list
        if re.match(r'^\d+\.\s', first_line):
            return 'numbered'
        
        # Quote
        if first_line.startswith('> '):
            return 'quote'
        
        # Table (has pipe separators)
        if '|' in stripped and '---' in stripped:
            return 'table'
        
        # Heading detection: **bold** + short + standalone line
        lines = stripped.split('\n')
        if len(lines) == 1 and len(first_line) < 100:
            # Check if wrapped in **bold** markers
            if first_line.startswith('**') and '**' in first_line[2:]:
                return 'heading'
            
            # Also detect by structure: short, title-like, all caps
            words = first_line.split()
            if words and (first_line.isupper() or sum(w[0].isupper() for w in words if w and w[0].isalpha()) > len(words) * 0.5):
                return 'heading'
        
        return 'paragraph'
    
    def _create_semantic_blocks_with_llm(self, page_md: str, page_num: int) -> List[Dict]:
        """Use LLM to group markdown into semantic blocks with full fidelity preservation.
        
        Returns list of block metadata with start_line, end_line, and rich content structure.
        """
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        prompt = f"""You are a **precise, deterministic document-structure transformer**.
Your only job is to convert input **Markdown** into a **BlockEditor-ready JSON block array** while keeping the document's **original formatting, structure, hierarchy, spacing, indentation, whitespace, tables, headers, inline styles, and layout EXACTLY preserved**.

This is a **mechanical conversion**, not an editing or rewriting task.
Do **not** improve grammar, fix OCR errors, merge lines, reorganize content, or change wording.

Your output must be **100% faithful to the original visual structure of the document**.

---

# CORE OBJECTIVE

Convert Markdown → BlockEditor JSON blocks.

* Maintain **1:1 fidelity** with the original OCR text.
* Preserve **all block boundaries exactly**.
* Preserve **all line breaks, spacing, indentation, and layout**.
* Preserve **all tables, lists, headings, inline marks, and structures**.
* Use **fallback preformatted blocks** when fidelity cannot be maintained through semantic types.

---

# BLOCK MAPPING RULES

Each logical unit from the Markdown becomes **one block**:

### 1. Headings
- # → heading (level: 1)
- ## → heading (level: 2)
- ### → heading (level: 3)
- #### → heading (level: 4)
- ##### → heading (level: 5)
- ###### → heading (level: 6)
- **Bold standalone line** → heading (detect level from context)

### 2. Paragraphs
- Normal text blocks → paragraph
- Preserve **exact line breaks and spacing** in content

### 3. Lists (IMPORTANT: Support nested structure)
- Lines starting with "- " → bulleted_list
- Lines starting with "1. ", "2. " → numbered_list
- Preserve indentation levels as nested children
- Output with "items" array containing {{content, children}}

### 4. Tables (IMPORTANT: Support rich structure)
- Markdown tables with | pipes → table
- Output with "columns" and "rows" arrays
- Preserve column count, order, and exact cell content

### 5. Preformatted / Fixed-Width Text
- Multi-space alignment → preformatted
- Forms with spacing → preformatted
- When semantic mapping loses fidelity → preformatted

### 6. Code Blocks
- Fenced code blocks → code
- Include "language" field if specified

### 7. Blockquotes
- Lines starting with "> " → blockquote

### 8. Special Elements
- Horizontal rules (---) → divider
- Images → image (with "src" and "alt" fields)
- Blank lines → empty

---

# STRICT PRESERVATION RULES

You MUST keep:

1. **All spacing** - every space, newline, and indent
2. **All indentation** - never collapse leading spaces
3. **OCR imperfections** - do NOT fix typos, broken hyphens, misspellings, extra newlines
4. **Block boundaries** - one logical component → one block
5. **Exact content** - no rewriting, merging, or reorganizing

---

# CONTENT CLEANING (IMPORTANT)

**STRIP ALL FORMATTING SYMBOLS** from the "content" field but **TRACK THEM AS METADATA**:

- "**Risk Policy**" → content: "Risk Policy", formatting: {{"bold": true}}
- "This is ==critical==" → content: "This is critical", formatting: {{"has_highlight": true}}
- "<small>footnote</small>" → content: "footnote", formatting: {{"size": "small"}}
- "<center>TITLE</center>" → content: "TITLE", formatting: {{"alignment": "center"}}

For inline formatting within paragraphs, use inline segments:
- "Hello **world**" → content: [{{"text": "Hello ", "bold": false}}, {{"text": "world", "bold": true}}]

---

# THINGS YOU MUST NOT DO

* ❌ Do NOT rewrite text
* ❌ Do NOT improve grammar
* ❌ Do NOT merge paragraphs
* ❌ Do NOT collapse blank lines
* ❌ Do NOT rearrange content
* ❌ Do NOT add, remove, or change words (except stripping formatting symbols)
* ❌ Do NOT fix OCR mistakes
* ❌ Do NOT convert preformatted text to paragraphs

---

# OUTPUT FORMAT

Return ONLY a JSON array (no other text):
[
  {{
    "start_line": 0,
    "end_line": 0,
    "content": "Risk Management Policy",
    "type": "heading",
    "level": 1,
    "formatting": {{"bold": true}}
  }},
  {{
    "start_line": 2,
    "end_line": 2,
    "content": "1. Overview",
    "type": "heading",
    "level": 2
  }},
  {{
    "start_line": 4,
    "end_line": 8,
    "content": "This paragraph has bold, italic, and highlighted words.",
    "type": "paragraph",
    "formatting": {{"has_bold": true, "has_italic": true, "has_highlight": true}}
  }},
  {{
    "start_line": 10,
    "end_line": 14,
    "type": "bulleted_list",
    "items": [
      {{
        "content": "Item 1",
        "children": [
          {{"content": "Sub-item 1.1"}},
          {{"content": "Sub-item 1.2"}}
        ]
      }},
      {{"content": "Item 2"}}
    ]
  }},
  {{
    "start_line": 16,
    "end_line": 20,
    "type": "table",
    "columns": ["Name", "Value"],
    "rows": [
      ["Risk Type", "Market Risk"],
      ["Severity", "High"]
    ]
  }},
  {{
    "start_line": 22,
    "end_line": 22,
    "content": "CONFIDENTIAL",
    "type": "paragraph",
    "formatting": {{"alignment": "center", "bold": true}}
  }},
  {{
    "start_line": 24,
    "end_line": 24,
    "content": "See section 3.1 for details.",
    "type": "paragraph",
    "formatting": {{"size": "small"}}
  }},
  {{
    "start_line": 26,
    "end_line": 28,
    "type": "code",
    "language": "python",
    "content": "def calculate_risk():\\n    return total * factor"
  }},
  {{
    "start_line": 30,
    "end_line": 30,
    "type": "blockquote",
    "content": "Important regulatory note"
  }}
]

BLOCK TYPES (use these exact strings):
- heading (with level 1-6): **bold** + short + standalone, or #/##/### markdown
- paragraph: normal text blocks
- bulleted_list: bullet lists with nested "items" array
- numbered_list: numbered lists with nested "items" array  
- table: markdown tables with "columns" and "rows" arrays
- blockquote: blockquotes (> text)
- callout: bordered/quoted sections (> Callout:)
- preformatted: multi-space aligned text, forms
- code: code blocks (with "language" field)
- divider: horizontal rules (---)
- image: images (with "src" and "alt" fields)
- empty: blank lines

FORMATTING METADATA (optional fields):
- "formatting": {{
    "bold": true,           // Entire block is bold
    "italic": true,         // Entire block is italic
    "has_bold": true,       // Contains some bold text
    "has_italic": true,     // Contains some italic text
    "has_highlight": true,  // Contains ==highlight==
    "alignment": "center|right|left",  // Text alignment
    "size": "small|normal|large"       // Font size
  }}
- "indent_level": 0-3      // For simple indentation (when not using nested structure)

HEADING LEVEL DETECTION:
- Level 1: # or first heading, all caps, or no numbering
- Level 2: ## or "1.", "2.", "3." (section numbers)
- Level 3: ### or "1.1", "2.1" (subsection numbers)
- Levels 4-6: ####, #####, ######

NESTED LISTS:
- Use "items" array with {{content, children}} structure
- Children is optional array of nested items
- Each level of indentation = one level of nesting

TABLES:
- Use "columns" array for header row
- Use "rows" array for data rows (each row is array of strings)
- Preserve exact cell content including spacing

TEXT TO ANALYZE (Page {page_num}):
{page_md}

Convert the above text to BlockEditor JSON blocks following ALL preservation rules.
Output ONLY valid JSON array. Strip ALL formatting symbols from content. Track formatting in metadata.
Use rich structures (items, columns, rows) for lists and tables.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and parse JSON
            response_text = response.content[0].text.strip()
            
            # Remove markdown code fences if present
            if response_text.startswith('```'):
                response_text = '\n'.join(response_text.split('\n')[1:-1])
            
            blocks = json.loads(response_text)
            
            self.logger.info(f"Page {page_num}: LLM created {len(blocks)} semantic blocks")
            return blocks
            
        except Exception as e:
            self.logger.error(f"Failed to create semantic blocks for page {page_num}: {e}")
            # Fallback: treat each line as a block
            lines = page_md.split('\n')
            return [
                {
                    "start_line": i,
                    "end_line": i,
                    "content": line,
                    "type": self._detect_block_type(line)
                }
                for i, line in enumerate(lines)
            ]
    
    def _extract_text_from_page(self, source_path: Path, page_num: int) -> str:
        """Extract raw text from a PDF page using pdfplumber"""
        try:
            import pdfplumber
        except ImportError:
            self.logger.warning("pdfplumber not installed, skipping text verification")
            return ""
        
        try:
            with pdfplumber.open(source_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    text = page.extract_text() or ""
                    return text
                return ""
        except Exception as e:
            self.logger.error(f"Failed to extract text from page {page_num}: {e}")
            return ""
    
    def _verify_page_with_text(self, vision_md: str, text_data: str, page_num: int, 
                                block_metadata: List[Dict]) -> List[Dict]:
        """Verify vision transcription against text extraction"""
        if not text_data.strip():
            self.logger.info(f"Page {page_num}: No text data for verification")
            return []

        if not self.client:
            return []

        # Format blocks for LLM with IDs
        blocks_for_llm = [
            {"block_id": b["id"], "content": b["content"]}
            for b in block_metadata
        ]

        prompt = f"""Compare vision transcription blocks with raw text extraction.

VISION TRANSCRIPTION (with block IDs):
{json.dumps(blocks_for_llm, indent=2)}

RAW TEXT (ground truth from PDF):
{text_data}

Find issues where vision transcription differs from raw text. For each issue:
1. Identify the EXACT block_id from the vision transcription above
2. Quote the original text from that block
3. Provide the corrected text from raw text
4. Explain the reason
5. Set confidence: "high" (certain error), "medium" (likely error), "low" (uncertain)

ONLY suggest corrections for:
- Typos or OCR errors (e.g., "Complance" → "Compliance")
- Missing text (e.g., missing footnotes, missing words)
- Wrong numbers or dates
- Misread special characters

Do NOT suggest corrections for:
- Formatting differences (markdown vs plain text)
- Line breaks or spacing
- Capitalization if both are valid
- Empty blocks (blocks with empty content)

Respond ONLY with a JSON array (no other text):
[
  {{
    "block_id": "p1_l5_a3f2b9",
    "original": "Complance",
    "suggested": "Compliance",
    "reason": "Typo - raw text shows 'Compliance'",
    "confidence": "high"
  }}
]

IMPORTANT: Use the EXACT block_id from the vision transcription list above.
If no issues found, return empty array: []
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text.strip()
            # Remove markdown code blocks if present
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            suggestions = json.loads(result_text)
            
            # Validate that block_ids exist
            valid_block_ids = {b['id'] for b in block_metadata}
            validated_suggestions = []
            
            for sug in suggestions:
                block_id = sug.get('block_id')
                if block_id in valid_block_ids:
                    validated_suggestions.append(sug)
                else:
                    self.logger.warning(f"Page {page_num}: Invalid block_id {block_id} in suggestion")
            
            if validated_suggestions:
                self.logger.info(f"Page {page_num}: Found {len(validated_suggestions)} verified suggestions")
            
            return validated_suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to verify page {page_num}: {e}")
            return []

    def _generate_table_of_contents(self, blocks: List[Dict]) -> List[Dict]:
        """Generate table of contents from heading blocks.
        
        Returns:
            List of TOC entries with title, level, block_id, page
        """
        toc = []
        for block in blocks:
            if block.get('type') == 'heading':
                toc.append({
                    'title': block['content'],
                    'level': block.get('level', 1),
                    'block_id': block['id'],
                    'page': block['page']
                })
        return toc

    def _convert_pdf_with_vision(self, source_path: Path) -> Tuple[str, List[Dict], List[Dict]]:
        """Convert PDF to markdown using vision-based transcription.
        
        Returns:
            Tuple of (markdown, block_metadata, verification_suggestions)
        """
        if not convert_from_path:
            raise RuntimeError("pdf2image is required but not installed")
        
        if not self.client:
            raise RuntimeError("Anthropic client not initialized - check ANTHROPIC_API_KEY")
        
        self.logger.info(f"Converting PDF to images at 300 DPI: {source_path}")
        
        # Convert PDF pages to images at 300 DPI
        images = convert_from_path(
            str(source_path),
            dpi=300,
            fmt='PNG',
            grayscale=False,
        )
        
        self.logger.info(f"Converted {len(images)} pages to images")
        
        # Transcribe each page and build block metadata
        all_blocks = []
        all_suggestions = []
        page_markdowns = []
        
        for page_num, image in enumerate(images, start=1):
            if self.use_direct_json:
                # DIRECT PDF → JSON: Bypass markdown entirely
                self.logger.info(f"Transcribing page {page_num}/{len(images)} directly to JSON...")
                page_result = self._transcribe_page_direct_to_json(image, page_num)
                
                # Extract blocks from result
                page_json_blocks = page_result.get('blocks', [])
                semantic_blocks = page_json_blocks  # Use directly
                
                # Build simple markdown for debugging/fallback
                page_md_lines = []
                for block in page_json_blocks:
                    content = block.get('content', '')
                    if isinstance(content, list):
                        text = ''.join(seg.get('text', '') for seg in content)
                    else:
                        text = str(content)
                    
                    if block.get('type') == 'heading':
                        level = block.get('level', 1)
                        page_md_lines.append('#' * level + ' ' + text)
                    elif block.get('type') == 'empty':
                        page_md_lines.append('')
                    else:
                        page_md_lines.append(text)
                
                page_md = '\n'.join(page_md_lines)
            else:
                # LEGACY: PDF → Markdown → JSON (2-step)
            self.logger.info(f"Transcribing page {page_num}/{len(images)}...")
            page_md = self._transcribe_page_with_vision(image, page_num)
            
            # Create semantic blocks using LLM
            self.logger.info(f"Creating semantic blocks for page {page_num}...")
            semantic_blocks = self._create_semantic_blocks_with_llm(page_md, page_num)
            
            # Generate stable IDs for each semantic block
            page_blocks = []
            for block_num, block_data in enumerate(semantic_blocks):
                # Use content for ID if available, otherwise use empty string
                content_for_id = block_data.get('content', '')
                if isinstance(content_for_id, list):
                    # If content is array of inline segments, use first segment
                    content_for_id = content_for_id[0].get('text', '') if content_for_id else ''
                
                block_id = self._generate_stable_block_id(page_num, block_num, str(content_for_id))
                
                # Start with required fields
                block_meta = {
                    'id': block_id,
                    'page': page_num,
                    'block_num': block_num,
                    'start_line': block_data.get('start_line', block_num),  # Direct JSON may not have line numbers
                    'end_line': block_data.get('end_line', block_num),
                    'type': block_data.get('type', 'paragraph')
                }
                
                # Add content field (can be string or array of inline segments)
                if 'content' in block_data:
                    block_meta['content'] = block_data['content']
                else:
                    block_meta['content'] = ''  # Default for blocks without content (like lists/tables)
                
                # Pass through ALL optional fields from LLM without filtering
                # This preserves rich structures like items, columns, rows, language, etc.
                optional_fields = [
                    'level', 'formatting', 'indent_level',  # Original fields
                    'items', 'columns', 'rows',  # Rich structures for lists and tables
                    'language',  # For code blocks
                    'src', 'alt',  # For images
                    'alignment', 'bbox',  # Direct JSON layout metadata
                ]
                for field in optional_fields:
                    if field in block_data:
                        block_meta[field] = block_data[field]
                
                all_blocks.append(block_meta)
                page_blocks.append(block_meta)
            
            # Verify with text extraction
            self.logger.info(f"Verifying page {page_num} with text extraction...")
            text_data = self._extract_text_from_page(source_path, page_num)
            page_suggestions = self._verify_page_with_text(page_md, text_data, page_num, page_blocks)
            all_suggestions.extend(page_suggestions)
            
            page_markdowns.append(page_md)
        
        # Combine all pages with page breaks
        full_markdown = "\n\n---\n\n".join(page_markdowns)
        
        # Generate table of contents from heading blocks
        toc = self._generate_table_of_contents(all_blocks)
        
        self.logger.info(f"Successfully transcribed {len(images)} pages")
        self.logger.info(f"Generated {len(all_blocks)} blocks with stable IDs")
        self.logger.info(f"Found {len(all_suggestions)} verification suggestions")
        self.logger.info(f"Generated TOC with {len(toc)} entries")
        
        return full_markdown, all_blocks, all_suggestions, toc

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_id = arguments["file_id"]
        source_path = resolve_path(arguments["source_path"])
        file_type = arguments["file_type"].lower().strip()

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # Initialize metadata
        block_metadata = []
        verification_suggestions = []
        table_of_contents = []

        # Handle different file types
        if file_type in self.SUPPORTED_TEXT_TYPES:
            markdown = self._convert_text_like(source_path)
            notes = "Converted from text-based source"
        elif file_type == "pdf":
            result = self._convert_pdf_with_vision(source_path)
            markdown, block_metadata, verification_suggestions, table_of_contents = result
            notes = "Converted from PDF using vision-based transcription with verification + TOC"
        else:
            markdown = f"> Conversion for file type `{file_type}` is not yet implemented.\n\n> Source file: `{source_path.name}`"
            notes = "Placeholder conversion"
            self.logger.warning(
                f"convert_to_markdown: unsupported file type file_id={file_id} type={file_type}"
            )

        # Save markdown
        md_file_id = generate_md_file_id(file_id)
        md_path = _MARKDOWN_OUTPUT_DIR / f"{md_file_id}.md"
        write_text_file(md_path, markdown)

        self.logger.info(
            f"convert_to_markdown: file_id={file_id} -> md_file_id={md_file_id} path={md_path}"
        )

        return {
            "file_id": file_id,
            "md_file_id": md_file_id,
            "md_path": str(md_path),
            "raw_markdown": markdown,
            "block_metadata": block_metadata,
            "verification_suggestions": verification_suggestions,
            "table_of_contents": table_of_contents,
            "notes": notes,
        }
