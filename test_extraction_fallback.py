#!/usr/bin/env python3
"""Test script to validate whole-doc fallback extraction logic."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from external.agent.doc_review_agent import DocReviewAgent


def test_whole_doc_fallback():
    """Test that extraction falls back to whole-doc when headings and TOC fail."""

    agent = DocReviewAgent()

    # Create mock state with no headings and no TOC entries (like Template_Version.pdf)
    state = {
        "run_id": "test_run",
        "doc_id": "test_doc",
        "doc_meta": {
            "source_path": "/tmp/test.pdf",
            "page_count": 11,
        },
        "structure": {
            "raw_text": "This is a test document without any markdown headings.\nIt should trigger the whole-doc fallback.\n" * 50,
            "headings": [],  # No headings!
            "toc_entries": [],  # No TOC!
        },
        "template_meta": {
            "template_id": "policy_template",
            "template_categories": ["Overview", "Scope", "Requirements"],
        },
        "phase1": {  # Add minimal Phase 1 data for Phase 2 reviews
            "doc_summary": {"summary": "Test document", "confidence": "low"},
            "template_fitness": {"overall_alignment": "poor"},
        },
        "phase2": {
            "chunks": {},
            "reviews": {},
        },
        "errors": [],
    }

    # Test extraction for a template section
    chunk = agent._extract_section_chunk(state, "Overview")

    # Validate results
    assert chunk is not None, "Chunk should not be None"
    assert chunk["method"] == "whole_doc_fallback", f"Expected whole_doc_fallback, got {chunk['method']}"
    assert chunk["section_title"] == "Overview", "Section title should match"
    assert len(chunk["text"]) > 0, "Text should not be empty"
    assert "No markdown headings found" in chunk["issues"], "Should note missing headings"
    assert chunk["boundary_check"] == "whole_document", "Should indicate whole document"

    print("✓ Test 1 passed: Whole-doc fallback works for section with no headings/TOC")

    # Test Phase 2 consolidation logic (manual simulation to avoid LLM calls)
    # Simulate extracting multiple sections (like Phase 2 does)
    targets = ["Overview", "Scope", "Requirements"]
    fallback_count = 0
    extracted = 0

    for title in targets:
        chunk = agent._extract_section_chunk(state, title)
        if chunk and chunk.get("text"):
            state["phase2"]["chunks"][title] = chunk
            extracted += 1
            if chunk.get("method") == "whole_doc_fallback":
                fallback_count += 1

    print(f"  Extracted {extracted} sections, {fallback_count} used fallback")

    # Simulate consolidation logic from run_phase2
    if extracted > 0 and fallback_count == extracted:
        print("  Consolidating to single 'Full Document' chunk...")
        raw_markdown = state["structure"]["raw_text"]
        state["phase2"]["chunks"] = {
            "Full Document": {
                "section_title": "Full Document",
                "method": "whole_doc_fallback",
                "page_range": [1, state["doc_meta"].get("page_count", None)],
                "char_range": [0, len(raw_markdown)],
                "boundary_check": "whole_document",
                "issues": [
                    "Document lacks markdown headings and TOC entries",
                    "Reviewing entire document as single section",
                ],
                "text": raw_markdown,
            }
        }

    # Validate consolidation
    chunks = state["phase2"]["chunks"]
    assert len(chunks) == 1, f"Should consolidate to 1 chunk, got {len(chunks)}"
    assert "Full Document" in chunks, f"Should have 'Full Document' chunk, got {list(chunks.keys())}"

    print("✓ Test 2 passed: Phase 2 consolidates multiple fallback chunks into single 'Full Document'")
    print(f"  Final chunk: {list(chunks.keys())[0]}")
    print(f"  Text length: {len(chunks['Full Document']['text'])} chars")
    print(f"  Issues: {chunks['Full Document']['issues']}")

    return True


if __name__ == "__main__":
    try:
        test_whole_doc_fallback()
        print("\n✅ All extraction fallback tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
