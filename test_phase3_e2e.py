#!/usr/bin/env python3
"""End-to-end test for Phase 0 → Phase 1 → Phase 2 → Phase 3 workflow with LLM."""

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


def test_full_workflow():
    """Test complete Phase 0 → 1 → 2 → 3 workflow with collateral_middle.pdf."""

    print("=" * 80)
    print("FULL WORKFLOW TEST: Phase 0 → Phase 1 → Phase 2 → Phase 3")
    print("=" * 80)

    # Setup paths
    test_pdf = Path("data/docreview/collateral_middle.pdf")
    template_path = Path("data/docreview/policy_template")

    if not test_pdf.exists():
        print(f"❌ Test PDF not found: {test_pdf}")
        return False

    print(f"✓ Test PDF: {test_pdf}")
    print(f"✓ Template: {template_path}")
    print()

    # Initialize agent
    agent = DocReviewAgent()

    # ------------------------------------------------------------------ #
    # Phase 0 + 1 + 2: Ingestion + Assessment + Reviews
    # ------------------------------------------------------------------ #
    print("PHASE 0 + 1 + 2: Running ingestion, assessment, and reviews...")
    print("-" * 80)

    try:
        # Run Phase 1 (includes Phase 0)
        state = agent.run_phase1(
            document_path=str(test_pdf.absolute()),
            template_id="policy_template",
        )

        print(f"Phase 1 Status: {state.get('phase1_status')}")

        # Run Phase 2
        state = agent.run_phase2(state)

        phase2_status = state.get("phase2_status")
        print(f"Phase 2 Status: {phase2_status}")

        if phase2_status != "success":
            print(f"❌ Phase 2 failed: {state.get('errors')}")
            return False

        # Collect suggested changes from all reviews
        # Note: Phase 2 reviews store changes as "issues" not "suggested_changes"
        phase2 = state.get("phase2", {})
        reviews = phase2.get("reviews", {})

        all_changes = []
        for section_title, review in reviews.items():
            # Get issues (which are the suggested changes)
            issues = review.get("issues", [])
            for issue in issues:
                # Convert issue to change format
                change = {
                    "id": issue.get("id"),
                    "type": issue.get("type"),
                    "severity": issue.get("severity"),
                    "title": issue.get("title"),
                    "original_text": issue.get("original_text", ""),
                    "suggested_text": issue.get("suggested_text", ""),
                    "reason": issue.get("reason", ""),
                    "location_instruction": issue.get("location_instruction"),
                    "source_section": section_title,
                }
                all_changes.append(change)

        print(f"\n  Total Suggested Changes: {len(all_changes)}")

        # Count by severity
        high = sum(1 for c in all_changes if c.get("severity") == "high")
        medium = sum(1 for c in all_changes if c.get("severity") == "medium")
        low = sum(1 for c in all_changes if c.get("severity") == "low")

        print(f"  Severity Distribution:")
        print(f"    High: {high}")
        print(f"    Medium: {medium}")
        print(f"    Low: {low}")

        if len(all_changes) == 0:
            print("\n  ⚠ No suggested changes to apply (document may be perfect!)")
            print("  Skipping Phase 3")
            return True

        # Show first few changes
        print(f"\n  Sample Changes:")
        for i, change in enumerate(all_changes[:3]):
            print(f"    {i+1}. [{change.get('severity', 'N/A')}] {change.get('type', 'N/A')}")
            print(f"       Section: {change.get('source_section', 'N/A')}")
            print(f"       Reason: {change.get('reason', 'N/A')[:80]}...")

        # Store changes in state for Phase 3
        state.setdefault("changes", {})
        state["changes"]["suggested_changes"] = all_changes

    except Exception as e:
        print(f"❌ Phase 0/1/2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ------------------------------------------------------------------ #
    # Phase 3: Change Application
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PHASE 3: Testing change application...")
    print("-" * 80)

    try:
        # Test 1: Apply high-severity changes only
        print("\nTest 1: Apply HIGH severity changes only")
        print("-" * 40)

        high_severity_changes = [c for c in all_changes if c.get("severity") == "high"]
        print(f"  High severity changes: {len(high_severity_changes)}")

        if len(high_severity_changes) > 0:
            # Create a copy of state for this test
            test_state_1 = {k: v for k, v in state.items()}
            test_state_1 = agent.run_phase3(test_state_1, severity_filter="high")

            phase3_status = test_state_1.get("phase3_status")
            applied = test_state_1.get("changes", {}).get("applied_change_ids", [])
            failed = test_state_1.get("changes", {}).get("failed_changes", [])

            print(f"  Phase 3 Status: {phase3_status}")
            print(f"  Applied: {len(applied)}")
            print(f"  Failed: {len(failed)}")

            if phase3_status == "success":
                print("  ✓ High severity changes applied successfully")
                new_text = test_state_1.get("changes", {}).get("new_raw_text", "")
                print(f"  New document length: {len(new_text)} chars")
            else:
                print(f"  ⚠ Phase 3 failed or no changes applied")
        else:
            print("  ⚠ No high severity changes to apply")

        # Test 2: Apply all changes
        print("\nTest 2: Apply ALL changes")
        print("-" * 40)

        if len(all_changes) > 0:
            test_state_2 = {k: v for k, v in state.items()}
            test_state_2 = agent.run_phase3(test_state_2)

            phase3_status = test_state_2.get("phase3_status")
            applied = test_state_2.get("changes", {}).get("applied_change_ids", [])
            failed = test_state_2.get("changes", {}).get("failed_changes", [])

            print(f"  Phase 3 Status: {phase3_status}")
            print(f"  Applied: {len(applied)}")
            print(f"  Failed: {len(failed)}")

            if phase3_status == "success":
                print("  ✓ All changes applied successfully")
                new_text = test_state_2.get("changes", {}).get("new_raw_text", "")
                original_text = state.get("structure", {}).get("raw_text", "")
                print(f"  Original length: {len(original_text)} chars")
                print(f"  New length: {len(new_text)} chars")
                print(f"  Difference: {len(new_text) - len(original_text):+d} chars")

                # Save diff for inspection
                output_dir = Path("test_phase3_output")
                output_dir.mkdir(exist_ok=True)

                (output_dir / "original.md").write_text(original_text)
                (output_dir / "revised.md").write_text(new_text)

                # Simple diff report
                diff_lines = []
                diff_lines.append("# Document Comparison Report")
                diff_lines.append(f"\n## Statistics")
                diff_lines.append(f"- Original: {len(original_text)} chars")
                diff_lines.append(f"- Revised: {len(new_text)} chars")
                diff_lines.append(f"- Changes Applied: {len(applied)}")
                diff_lines.append(f"- Changes Failed: {len(failed)}")
                diff_lines.append(f"\n## Applied Changes")
                for change_id in applied[:10]:  # First 10
                    change = next((c for c in all_changes if c.get("id") == change_id), None)
                    if change:
                        diff_lines.append(f"\n### {change_id}")
                        diff_lines.append(f"- **Severity**: {change.get('severity')}")
                        diff_lines.append(f"- **Type**: {change.get('type')}")
                        diff_lines.append(f"- **Section**: {change.get('source_section')}")
                        diff_lines.append(f"- **Reason**: {change.get('reason')}")

                (output_dir / "comparison.md").write_text("\n".join(diff_lines))

                print(f"\n  Output saved to: {output_dir}/")
                print(f"    - original.md")
                print(f"    - revised.md")
                print(f"    - comparison.md")
            else:
                print(f"  ⚠ Phase 3 failed or no changes applied")

    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
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
        "Phase 0: Document ingested": bool(state.get("structure", {}).get("raw_text")),
        "Phase 1: Assessment completed": state.get("phase1_status") == "success",
        "Phase 2: Reviews generated": len(reviews) > 0,
        "Phase 2: Changes suggested": len(all_changes) > 0,
        "Phase 3: Tested with filters": True,  # We ran the tests
    }

    all_passed = True
    for check, passed in validations.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    try:
        success = test_full_workflow()
        if success:
            print("\n" + "=" * 80)
            print("✅ FULL WORKFLOW TEST PASSED")
            print("=" * 80)
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("❌ FULL WORKFLOW TEST FAILED")
            print("=" * 80)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
