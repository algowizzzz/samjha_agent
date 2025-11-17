#!/usr/bin/env python3
"""Inspect Phase 2 review output to see what's being generated."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from external.agent.doc_review_agent import DocReviewAgent


def inspect_reviews():
    """Run Phase 2 and inspect review details."""

    print("=" * 80)
    print("REVIEW INSPECTION: Phase 2 Output Analysis")
    print("=" * 80)

    test_pdf = Path("data/docreview/collateral_middle.pdf")
    agent = DocReviewAgent()

    # Run Phase 1 + 2
    state = agent.run_phase1(
        document_path=str(test_pdf.absolute()),
        template_id="policy_template",
    )
    state = agent.run_phase2(state)

    # Inspect reviews
    reviews = state.get("phase2", {}).get("reviews", {})

    print(f"\nTotal Reviews Generated: {len(reviews)}")
    print("=" * 80)

    for section_title, review in reviews.items():
        print(f"\n[{section_title}]")
        print("-" * 80)

        # Show review structure
        print(f"  Keys in review: {list(review.keys())}")

        # Check for suggested changes
        changes = review.get("suggested_changes", [])
        print(f"  Suggested Changes: {len(changes)}")

        if len(changes) > 0:
            for i, change in enumerate(changes):
                print(f"\n  Change {i+1}:")
                print(f"    ID: {change.get('id', 'N/A')}")
                print(f"    Type: {change.get('type', 'N/A')}")
                print(f"    Severity: {change.get('severity', 'N/A')}")
                print(f"    Reason: {change.get('reason', 'N/A')[:100]}...")
                print(f"    Original: {change.get('original_text', 'N/A')[:50]}...")
                print(f"    Suggested: {change.get('suggested_text', 'N/A')[:50]}...")

        # Show other review fields
        for key, value in review.items():
            if key != "suggested_changes":
                if isinstance(value, str):
                    print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
                elif isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")

    # Check summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    summary = state.get("phase2", {}).get("summary_report", {})
    if summary:
        print(json.dumps(summary, indent=2))
    else:
        print("  No summary report generated")

    # Save full state for inspection
    output_file = Path("test_inspect_reviews_output.json")
    with open(output_file, "w") as f:
        # Save reviews only
        json.dump({
            "reviews": reviews,
            "summary": summary,
        }, f, indent=2, default=str)

    print(f"\n  Full output saved to: {output_file}")


if __name__ == "__main__":
    inspect_reviews()
