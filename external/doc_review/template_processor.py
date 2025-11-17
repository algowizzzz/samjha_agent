"""
Template-based document improvement processor.
Performs gap analysis and content improvement using templates.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class TemplateProcessor:
    """Process documents against templates with LLM-based gap analysis and improvement."""
    
    def __init__(self, api_key: str):
        """Initialize with Anthropic API key."""
        self.client = Anthropic(api_key=api_key)
        self.config_dir = Path(__file__).parent.parent.parent / "config" / "prompts" / "doc-review"
        
        # Load prompts
        self.gap_analysis_prompt = self._load_prompt("gap_analysis.txt")
        self.content_improvement_prompt = self._load_prompt("content_improvement.txt")
        
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from config."""
        prompt_path = self.config_dir / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        return prompt_path.read_text()
    
    def process_document_with_template(
        self,
        full_markdown: str,
        block_metadata: List[Dict[str, Any]],
        template_content: str,
        template_name: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process entire document against template page by page.
        
        Args:
            full_markdown: Complete document markdown
            block_metadata: List of block metadata with IDs
            template_content: Template markdown content
            template_name: Name of template being applied
            
        Returns:
            Tuple of (gap_analysis_results, improvement_suggestions)
        """
        logger.info(f"Starting template processing with '{template_name}'")
        
        # Group blocks by page
        pages = self._group_blocks_by_page(block_metadata, full_markdown)
        
        all_gap_analysis = []
        all_improvements = []
        new_doc_so_far = ""
        
        for page_num, page_data in enumerate(pages, start=1):
            logger.info(f"Processing page {page_num}/{len(pages)}")
            
            # Step 1: Gap Analysis
            gap_analysis = self._perform_gap_analysis(
                full_document=full_markdown,
                template=template_content,
                new_doc_so_far=new_doc_so_far,
                current_page=page_data['markdown'],
                page_blocks=page_data['blocks'],
                page_num=page_num
            )
            all_gap_analysis.extend(gap_analysis)
            
            # Step 2: Content Improvement
            improvements = self._generate_improvements(
                full_document=full_markdown,
                template=template_content,
                new_doc_so_far=new_doc_so_far,
                current_page=page_data['markdown'],
                page_blocks=page_data['blocks'],
                gap_analysis=gap_analysis,
                page_num=page_num
            )
            all_improvements.extend(improvements)
            
            # Update new_doc_so_far with improved content
            improved_page = self._apply_improvements_to_page(
                page_data['markdown'],
                improvements
            )
            new_doc_so_far += improved_page + "\n\n---\n\n"
        
        logger.info(f"Template processing complete: {len(all_gap_analysis)} gaps, {len(all_improvements)} improvements")
        return all_gap_analysis, all_improvements
    
    def _group_blocks_by_page(
        self,
        block_metadata: List[Dict[str, Any]],
        full_markdown: str
    ) -> List[Dict[str, Any]]:
        """Group blocks by page number."""
        pages = {}
        markdown_lines = full_markdown.split('\n')
        
        for block in block_metadata:
            page_num = block['page']
            if page_num not in pages:
                pages[page_num] = {
                    'blocks': [],
                    'markdown': ''
                }
            pages[page_num]['blocks'].append(block)
        
        # Extract markdown for each page
        for page_num in pages:
            page_blocks = pages[page_num]['blocks']
            if page_blocks:
                # Get markdown from first to last line of page
                start_line = min(b['start_line'] for b in page_blocks)
                end_line = max(b['end_line'] for b in page_blocks)
                pages[page_num]['markdown'] = '\n'.join(markdown_lines[start_line:end_line+1])
        
        # Return sorted by page number
        return [pages[p] for p in sorted(pages.keys())]
    
    def _perform_gap_analysis(
        self,
        full_document: str,
        template: str,
        new_doc_so_far: str,
        current_page: str,
        page_blocks: List[Dict],
        page_num: int
    ) -> List[Dict]:
        """Perform gap analysis for a single page."""
        logger.info(f"Performing gap analysis for page {page_num}")
        
        # Prepare blocks for LLM
        blocks_json = json.dumps([
            {
                'block_id': b['id'],
                'content': b['content'],
                'type': b['type']
            }
            for b in page_blocks
        ], indent=2)
        
        user_prompt = f"""
ORIGINAL FULL DOCUMENT:
{full_document}

---

TEMPLATE:
{template}

---

NEW DOCUMENT SO FAR:
{new_doc_so_far if new_doc_so_far else "(This is page 1, no previous content)"}

---

CURRENT PAGE (Page {page_num}):
{current_page}

---

BLOCK METADATA:
{blocks_json}

---

Analyze the current page against the template and identify gaps block by block.
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                system=self.gap_analysis_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Remove markdown code fences if present
            if response_text.startswith('```'):
                response_text = '\n'.join(response_text.split('\n')[1:-1])
            
            result = json.loads(response_text)
            gap_analysis = result.get('gap_analysis', [])
            
            logger.info(f"Page {page_num}: Found {len(gap_analysis)} gaps")
            return gap_analysis
            
        except Exception as e:
            logger.error(f"Gap analysis failed for page {page_num}: {e}")
            return []
    
    def _generate_improvements(
        self,
        full_document: str,
        template: str,
        new_doc_so_far: str,
        current_page: str,
        page_blocks: List[Dict],
        gap_analysis: List[Dict],
        page_num: int
    ) -> List[Dict]:
        """Generate content improvements for a single page."""
        logger.info(f"Generating improvements for page {page_num}")
        
        # Prepare blocks for LLM
        blocks_json = json.dumps([
            {
                'block_id': b['id'],
                'content': b['content'],
                'type': b['type']
            }
            for b in page_blocks
        ], indent=2)
        
        gap_analysis_json = json.dumps(gap_analysis, indent=2)
        
        user_prompt = f"""
ORIGINAL FULL DOCUMENT:
{full_document}

---

TEMPLATE:
{template}

---

NEW DOCUMENT SO FAR:
{new_doc_so_far if new_doc_so_far else "(This is page 1, no previous content)"}

---

CURRENT PAGE (Page {page_num}):
{current_page}

---

BLOCK METADATA:
{blocks_json}

---

GAP ANALYSIS:
{gap_analysis_json}

---

Generate improved content for each block that addresses the identified gaps.
"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                system=self.content_improvement_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Remove markdown code fences if present
            if response_text.startswith('```'):
                response_text = '\n'.join(response_text.split('\n')[1:-1])
            
            result = json.loads(response_text)
            improvements = result.get('improvements', [])
            
            # Validate improvements match blocks
            valid_improvements = []
            block_ids = {b['id'] for b in page_blocks}
            for imp in improvements:
                if imp.get('block_id') in block_ids:
                    valid_improvements.append(imp)
                else:
                    logger.warning(f"Improvement has invalid block_id: {imp.get('block_id')}")
            
            logger.info(f"Page {page_num}: Generated {len(valid_improvements)} improvements")
            return valid_improvements
            
        except Exception as e:
            logger.error(f"Content improvement failed for page {page_num}: {e}")
            return []
    
    def _apply_improvements_to_page(
        self,
        page_markdown: str,
        improvements: List[Dict]
    ) -> str:
        """Apply improvements to page markdown (for context in next page)."""
        # For now, just return original - improvements are shown as suggestions
        # In future, could optionally auto-apply improvements
        return page_markdown


def load_template(template_name: str) -> str:
    """Load template content by name."""
    template_dir = Path(__file__).parent.parent.parent / "data" / "templates"
    template_path = template_dir / f"{template_name}.md"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_name}")
    
    return template_path.read_text()


def list_templates() -> List[str]:
    """List available template names."""
    template_dir = Path(__file__).parent.parent.parent / "data" / "templates"
    if not template_dir.exists():
        return []
    
    templates = []
    for path in template_dir.glob("*.md"):
        templates.append(path.stem)
    
    return sorted(templates)

