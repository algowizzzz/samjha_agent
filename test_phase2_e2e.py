#!/usr/bin/env python3
"""End-to-end test for Phase 0 → Phase 1 → Phase 2 workflow with real PDFs."""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from external.agent.doc_review_agent import DocReviewAgent


def test_phase_0_1_2_e2e():
    """Test complete Phase 0 → 1 → 2 workflow with collateral_middle.pdf."""

    print("=" * 80)
    print("END-TO-END TEST: Phase 0 → Phase 1 → Phase 2")
    print("=" * 80)

    # Setup paths
    test_pdf = Path("data/docreview/collateral_middle.pdf")
    template_path = Path("data/docreview/policy_template")

    if not test_pdf.exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        return False

    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        return False

    print(f"✓ Test PDF: {test_pdf}")
    print(f"✓ Template: {template_path}")
    print()

    # Initialize agent
    agent = DocReviewAgent()

    # ------------------------------------------------------------------ #
    # Phase 0 + 1: Ingestion + Holistic Assessment
    # ------------------------------------------------------------------ #
    print("PHASE 0 + 1: Running ingestion and holistic assessment...")
    print("-" * 80)

    try:
        state = agent.run_phase1(
            document_path=str(test_pdf.absolute()),
            template_id="policy_template",
        )

        # Validate Phase 0 outputs (ingestion)
        print("\n[Phase 0] Document Ingestion:")
        doc_meta = state.get("doc_meta", {})
        structure = state.get("structure", {})

        print(f"  File Type: {doc_meta.get('file_type', 'unknown')}")
        print(f"  Page Count: {doc_meta.get('page_count', 0)}")
        print(f"  Word Count: {structure.get('file_stats', {}).get('word_count', 0)}")
        print(f"  Raw Text Length: {len(structure.get('raw_text', ''))} chars")
        print(f"  Headings Found: {len(structure.get('headings', []))}")
        print(f"  TOC Entries: {len(structure.get('toc_entries', []))}")

        assert structure.get("raw_text"), "❌ No raw text extracted"
        print("  ✓ Document ingested successfully")

        # Validate Phase 1 outputs (if LLM available)
        print("\n[Phase 1] Holistic Assessment:")
        phase1 = state.get("phase1", {})
        phase1_status = state.get("phase1_status", "unknown")
        print(f"  Status: {phase1_status}")

        if phase1_status == "success":
            doc_summary = phase1.get("doc_summary", {})
            print(f"  Summary: {doc_summary.get('summary', 'N/A')[:100]}...")
            print(f"  Confidence: {doc_summary.get('confidence', 'N/A')}")
            template_fitness = phase1.get("template_fitness", {})
            print(f"  Template Alignment: {template_fitness.get('overall_alignment', 'N/A')}")
            print("  ✓ Phase 1 completed with LLM")
        else:
            print("  ⚠ Phase 1 skipped (no LLM available)")
            # Add minimal Phase 1 data for Phase 2 to work
            state["phase1"] = {
                "doc_summary": {"summary": "Test document", "confidence": "low"},
                "template_fitness": {"overall_alignment": "unknown"},
            }

    except Exception as e:
        print(f"❌ Phase 0/1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------ #
    # Phase 2: Section Extraction + Reviews
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 2: Running section extraction and reviews...")
    print("-" * 80)

    try:
        # Run Phase 2
        state = agent.run_phase2(state)

        phase2_status = state.get("phase2_status", "unknown")
        phase2 = state.get("phase2", {})
        chunks = phase2.get("chunks", {})
        reviews = phase2.get("reviews", {})

        print(f"\n[Phase 2] Section Extraction:")
        print(f"  Status: {phase2_status}")
        print(f"  Chunks Extracted: {len(chunks)}")

        if phase2_status == "failed":
            errors = state.get("errors", [])
            print(f"  ❌ Phase 2 failed: {errors}")
            return False

        # Validate chunks
        assert len(chunks) > 0, "❌ No chunks extracted"
        print("  ✓ Chunks extracted successfully")

        # Show chunk details
        print(f"\n  Chunk Details:")
        for title, chunk in chunks.items():
            method = chunk.get("method", "unknown")
            text_len = len(chunk.get("text", ""))
            issues = chunk.get("issues", [])
            print(f"    - {title}:")
            print(f"        Method: {method}")
            print(f"        Text Length: {text_len} chars")
            if issues:
                print(f"        Issues: {', '.join(issues[:2])}...")

        # Validate reviews (if LLM available)
        print(f"\n[Phase 2] Section Reviews:")
        print(f"  Reviews Generated: {len(reviews)}")

        if len(reviews) > 0:
            print("  ✓ Reviews generated with LLM")
            # Show review details
            for section_title, review in list(reviews.items())[:3]:  # First 3
                print(f"    - {section_title}:")
                suggestions = review.get("suggested_changes", [])
                print(f"        Suggested Changes: {len(suggestions)}")
                if suggestions:
                    high = sum(1 for s in suggestions if s.get("severity") == "high")
                    medium = sum(1 for s in suggestions if s.get("severity") == "medium")
                    low = sum(1 for s in suggestions if s.get("severity") == "low")
                    print(f"        Severity: {high} high, {medium} medium, {low} low")
        else:
            print("  ⚠ Reviews skipped (no LLM available)")

        # Validate summary report
        summary = phase2.get("summary_report", {})
        if summary:
            print(f"\n  Summary Report:")
            print(f"    Total Issues: {summary.get('total_issues', 0)}")
            print(f"    Critical Sections: {len(summary.get('critical_sections', []))}")
            print("  ✓ Summary report generated")

    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------ #
    # Validation Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    validations = {
        "Phase 0: Document ingested": bool(structure.get("raw_text")),
        "Phase 0: Headings parsed": "headings" in structure,
        "Phase 0: Metadata extracted": "file_type" in doc_meta,
        "Phase 1: Structure available": phase1_status in ["success", "failed", "skipped"],
        "Phase 2: Chunks extracted": len(chunks) > 0,
        "Phase 2: Completed successfully": phase2_status == "success",
        "Phase 2: At least one chunk": len(chunks) >= 1,
    }

    all_passed = True
    for check, passed in validations.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False

    # Output state for inspection
    output_file = Path("test_phase2_e2e_output.json")
    with open(output_file, "w") as f:
        # Clean state for JSON serialization
        clean_state = {
            "run_id": state.get("run_id"),
            "doc_id": state.get("doc_id"),
            "doc_meta": doc_meta,
            "phase1_status": phase1_status,
            "phase2_status": phase2_status,
            "chunks_count": len(chunks),
            "reviews_count": len(reviews),
            "chunk_titles": list(chunks.keys()),
        }
        json.dump(clean_state, f, indent=2)

    print(f"\n  State snapshot saved to: {output_file}")

    return all_passed


if __name__ == "__main__":
    try:
        success = test_phase_0_1_2_e2e()
        if success:
            print("\n" + "=" * 80)
            print("✅ END-TO-END TEST PASSED")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ END-TO-END TEST FAILED")
            print("=" * 80)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
