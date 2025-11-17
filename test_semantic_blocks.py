#!/usr/bin/env python3
"""Test semantic block creation with vision-based PDF conversion."""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from external.tools.doc_processing.convert_to_markdown import ConvertToMarkdownTool

def test_semantic_blocks():
    """Test semantic block creation on collateral_middle.pdf"""
    
    # Set API key
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', '')
    
    # Initialize tool
    tool = ConvertToMarkdownTool()
    
    # Test file
    test_file = Path("data/docreview/collateral_middle.pdf")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"🔍 Testing semantic block creation on: {test_file}")
    print("=" * 80)
    
    # Execute conversion
    result = tool.execute({
        "file_id": "test_semantic_blocks",
        "source_path": str(test_file),
        "file_type": "pdf"
    })
    
    # Extract results
    markdown = result['raw_markdown']
    block_metadata = result['block_metadata']
    suggestions = result['verification_suggestions']
    
    print(f"\n✅ Conversion complete!")
    print(f"   - Total blocks: {len(block_metadata)}")
    print(f"   - Verification suggestions: {len(suggestions)}")
    
    # Save markdown
    output_md = Path("test_semantic_blocks_output.md")
    output_md.write_text(markdown, encoding='utf-8')
    print(f"\n📄 Markdown saved to: {output_md}")
    
    # Save block metadata
    output_blocks = Path("test_semantic_blocks_metadata.json")
    output_blocks.write_text(json.dumps(block_metadata, indent=2), encoding='utf-8')
    print(f"📦 Block metadata saved to: {output_blocks}")
    
    # Save suggestions
    output_suggestions = Path("test_semantic_blocks_suggestions.json")
    output_suggestions.write_text(json.dumps(suggestions, indent=2), encoding='utf-8')
    print(f"💡 Suggestions saved to: {output_suggestions}")
    
    # Print sample blocks
    print("\n" + "=" * 80)
    print("📊 SAMPLE BLOCKS (first 5):")
    print("=" * 80)
    for i, block in enumerate(block_metadata[:5]):
        print(f"\nBlock {i+1}:")
        print(f"  ID: {block['id']}")
        print(f"  Type: {block['type']}")
        print(f"  Page: {block['page']}, Block #: {block['block_num']}")
        print(f"  Lines: {block['start_line']} - {block['end_line']}")
        print(f"  Content: {block['content'][:100]}..." if len(block['content']) > 100 else f"  Content: {block['content']}")
    
    # Print sample suggestions
    if suggestions:
        print("\n" + "=" * 80)
        print("💡 SAMPLE SUGGESTIONS (first 3):")
        print("=" * 80)
        for i, sug in enumerate(suggestions[:3]):
            print(f"\nSuggestion {i+1}:")
            print(f"  Block ID: {sug['block_id']}")
            print(f"  Original: {sug['original'][:80]}...")
            print(f"  Suggested: {sug['suggested'][:80]}...")
            print(f"  Reason: {sug['reason']}")
            print(f"  Confidence: {sug['confidence']}")
    
    print("\n" + "=" * 80)
    print("✅ Test complete! Review the output files.")
    print("=" * 80)

if __name__ == "__main__":
    test_semantic_blocks()

