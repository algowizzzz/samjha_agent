#!/usr/bin/env python3
"""Test what Claude Haiku produces from the semantic block prompt"""

import json
import os
from anthropic import Anthropic

# Sample markdown from the actual document (first 10 lines)
test_markdown = """**Guideline**

**Subject:** Capital Adequacy Requirements (CAR)

**Chapter 1 – Overview of risk-based capital requirements**

**Effective Date:** November 2023 / January 2024 

**Note:** For institutions with a fiscal year ending October 31 or December 31, respectively.

Subsections 485(1) and 949(1) of the **Bank Act (BA)**, subsection 473(1) of the **Trust and Loan Companies Act (TLCA)** require banks (including federal credit unions), bank holding companies, federally regulated trust companies, and federally regulated loan companies to maintain adequate capital."""

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

# CONTENT CLEANING (CRITICAL RULE)

**STRIP ALL FORMATTING SYMBOLS** from the "content" field but **TRACK THEM AS METADATA**:

Examples:
- Input: "**Risk Policy**"
  Output: {{"content": "Risk Policy", "type": "heading", "level": 1, "formatting": {{"bold": true}}}}

- Input: "**Subject:** Capital Requirements" 
  Output: {{"content": "Subject: Capital Requirements", "type": "heading", "level": 2, "formatting": {{"bold": true}}}}

- Input: "The **Bank Act (BA)** requires..."
  Output: {{"content": [{{"text": "The "}}, {{"text": "Bank Act (BA)", "bold": true}}, {{"text": " requires..."}}], "type": "paragraph"}}

KEY RULE: content field MUST NOT contain ** or __ or * or _ symbols. Strip them and put bold/italic info in formatting or inline segments.

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
    "content": [{{"text": "The "}}, {{"text": "important", "bold": true}}, {{"text": " text"}}],
    "type": "paragraph"
  }}
]

BLOCK TYPES: heading, paragraph, bulleted_list, numbered_list, table, blockquote, preformatted, code, divider, image, empty

TEXT TO ANALYZE (Page 1):
{test_markdown}

Convert the above text to BlockEditor JSON blocks.
CRITICAL: Strip ALL ** and * symbols from content. Track bold/italic in formatting metadata or inline segments.
Output ONLY valid JSON array.
"""

def test_llm_output():
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    print("=" * 80)
    print("TESTING CLAUDE HAIKU WITH SEMANTIC BLOCK PROMPT")
    print("=" * 80)
    print("\nINPUT MARKDOWN:")
    print("-" * 80)
    print(test_markdown)
    print("-" * 80)
    
    print("\nCALLING CLAUDE HAIKU...")
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = response.content[0].text.strip()
    
    print("\nRAW RESPONSE:")
    print("-" * 80)
    print(response_text)
    print("-" * 80)
    
    # Remove markdown code fences if present
    if response_text.startswith('```'):
        response_text = '\n'.join(response_text.split('\n')[1:-1])
    
    try:
        blocks = json.loads(response_text)
        print("\nPARSED JSON (first 5 blocks):")
        print("-" * 80)
        for i, block in enumerate(blocks[:5]):
            print(f"\n{i+1}. Type: {block.get('type')}")
            print(f"   Content: {block.get('content')}")
            print(f"   Formatting: {block.get('formatting', 'None')}")
            print(f"   Level: {block.get('level', 'None')}")
        print("-" * 80)
        
        # Check for issues
        print("\n🔍 CHECKING FOR ISSUES:")
        print("-" * 80)
        issues = []
        for i, block in enumerate(blocks):
            content = block.get('content', '')
            if isinstance(content, str) and ('**' in content or '__' in content):
                issues.append(f"Block {i}: Still has ** or __ in content: {content[:50]}")
            if block.get('type') == 'heading' and not block.get('level'):
                issues.append(f"Block {i}: Heading missing 'level' field")
            if block.get('type') == 'heading' and not block.get('formatting'):
                issues.append(f"Block {i}: Heading missing 'formatting' metadata")
        
        if issues:
            print("❌ FOUND ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ No formatting symbols found in content fields")
            print("✅ All headings have level and formatting metadata")
        
    except json.JSONDecodeError as e:
        print(f"\n❌ FAILED TO PARSE JSON: {e}")

if __name__ == "__main__":
    test_llm_output()

