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
        """Use LLM to group markdown into semantic blocks (paragraphs, headings, lists, tables).
        
        Returns list of block metadata with start_line, end_line, and content.
        """
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        prompt = f"""Analyze this text and group it into semantic blocks with ALL formatting metadata.

TEXT TO ANALYZE:
{page_md}

INSTRUCTIONS:
- Group related lines into semantic blocks (entire paragraphs, headings, lists, tables)
- Each block should be a complete unit
- DO NOT split paragraphs or lists into multiple blocks
- Empty lines can be their own blocks
- Detect headings by **bold** formatting + structure
- Infer heading level (1-3) from context
- STRIP markdown symbols (**bold**, *italic*, ==highlight==, <small>, <center>, <right>) from content field
- Store ALL formatting info as metadata fields

OUTPUT FORMAT (JSON array with formatting metadata):
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
    "end_line": 12,
    "content": "- Item 1\\n  - Sub-item 1.1\\n  - Sub-item 1.2",
    "type": "bullet",
    "indent_level": 2
  }},
  {{
    "start_line": 14,
    "end_line": 14,
    "content": "CONFIDENTIAL",
    "type": "paragraph",
    "formatting": {{"alignment": "center", "bold": true}}
  }},
  {{
    "start_line": 16,
    "end_line": 16,
    "content": "See section 3.1 for details.",
    "type": "paragraph",
    "formatting": {{"size": "small"}}
  }},
  {{
    "start_line": 18,
    "end_line": 20,
    "content": "Important: Review annually.",
    "type": "callout"
  }}
]

CRITICAL CONTENT CLEANING:
- Strip ALL formatting symbols from content field
- "**Risk Policy**" → "Risk Policy"
- "This is ==critical==" → "This is critical"
- "<small>footnote</small>" → "footnote"
- "<center>TITLE</center>" → "TITLE"
- But track what was stripped in the "formatting" object

BLOCK TYPES:
- heading (with level 1-3): **bold** + short + standalone
- paragraph: multi-line text blocks
- bullet: bullet lists (lines with -)
- numbered: numbered lists (1. 2. 3.)
- table: markdown tables with | pipes
- callout: bordered/quoted sections (> Callout:)
- quote: blockquotes (> text)
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
- "indent_level": 0-3      // For nested lists (spaces/2)

HEADING LEVEL DETECTION:
- Level 1: First heading, all caps, or no numbering
- Level 2: "1.", "2.", "3." (section numbers)
- Level 3: "1.1", "2.1" (subsection numbers)

CRITICAL: Output ONLY valid JSON array. Strip ALL symbols from content. Track formatting in metadata.
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
            self.logger.info(f"Transcribing page {page_num}/{len(images)}...")
            page_md = self._transcribe_page_with_vision(image, page_num)
            
            # Create semantic blocks using LLM
            self.logger.info(f"Creating semantic blocks for page {page_num}...")
            semantic_blocks = self._create_semantic_blocks_with_llm(page_md, page_num)
            
            # Generate stable IDs for each semantic block
            page_blocks = []
            for block_num, block_data in enumerate(semantic_blocks):
                block_id = self._generate_stable_block_id(page_num, block_num, block_data['content'])
                block_meta = {
                    'id': block_id,
                    'page': page_num,
                    'block_num': block_num,
                    'start_line': block_data['start_line'],
                    'end_line': block_data['end_line'],
                    'content': block_data['content'],
                    'type': block_data['type']
                }
                
                # Add optional metadata fields if present
                if block_data.get('level'):
                    block_meta['level'] = block_data['level']
                if block_data.get('formatting'):
                    block_meta['formatting'] = block_data['formatting']
                if block_data.get('indent_level') is not None:
                    block_meta['indent_level'] = block_data['indent_level']
                
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
