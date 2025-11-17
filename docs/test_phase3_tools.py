#!/usr/bin/env python3
"""
Test script for Phase 3 LLM-driven template mapping & refinement.
Tests each LLM node individually and compares against expected outputs.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv not available, try to load .env manually
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from external.agent.doc_review_agent import DocReviewAgent
from external.doc_review.llm import LLMNotAvailableError


# Test configuration
BASE_DIR = Path(__file__).parent
PHASE2_RESULTS_DIR = BASE_DIR / "test_phase2_results"
RESULTS_DIR = BASE_DIR / "test_phase3_results"
EXPECTED_FILE = BASE_DIR / "test_phase3_expected.json"

# Load Phase 2 results for input to Phase 3 nodes
PHASE2_INDEX_FILE = PHASE2_RESULTS_DIR / "build_index" / "output.json"
PHASE2_TEMPLATE_FILE = PHASE2_RESULTS_DIR / "load_outline_template" / "output.json"
PHASE2_CHUNKS_FILE = PHASE2_RESULTS_DIR / "chunk_markdown" / "output.json"

FILE_ID = "car-chapter-1-overview"


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def save_result(tool_name: str, result: Dict[str, Any]) -> Path:
    """Save tool result to JSON file."""
    tool_dir = RESULTS_DIR / tool_name
    ensure_dir(tool_dir)
    output_file = tool_dir / "output.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved output to: {output_file}")
    return output_file


def load_phase2_results() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Load Phase 2 results needed for Phase 3 tests."""
    with PHASE2_INDEX_FILE.open("r", encoding="utf-8") as f:
        index_result = json.load(f)
    
    with PHASE2_CHUNKS_FILE.open("r", encoding="utf-8") as f:
        chunks_result = json.load(f)
    
    index = index_result.get("index", {})
    chunks = chunks_result.get("chunks", [])
    
    # Load template - try from Phase 2 results first, otherwise load directly
    template = {}
    if PHASE2_TEMPLATE_FILE.exists():
        with PHASE2_TEMPLATE_FILE.open("r", encoding="utf-8") as f:
            template_result = json.load(f)
            template = template_result.get("template", {})
    else:
        # Load template directly using the tool
        from external.tools.doc_processing.load_outline_template import LoadOutlineTemplateTool
        try:
            tool = LoadOutlineTemplateTool()
            # Use policy_template as default template
            template_result = tool.execute({"template_id": "policy_template"})
            template = template_result.get("template", {})
            print(f"  Loaded template 'policy_template' directly: {len(template.get('sections', []))} sections")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load template: {e}")
            # Create minimal template structure for testing
            template = {
                "sections": [
                    {"section_id": "intro", "title": "Introduction"},
                    {"section_id": "overview", "title": "Overview"},
                    {"section_id": "definitions", "title": "Definitions"}
                ]
            }
    
    # Build minimal state for agent nodes
    state = {
        "file_id": FILE_ID,
        "index": index,
        "template": template,
        "chunks": chunks,
        "config": {
            "template": {
                "template_id": "policy_template",
                "late_append_relevancy_threshold": 0.4
            }
        },
        "logs": []
    }
    
    return state, index, template, chunks


def normalize_for_comparison(value: Any, path: str = "") -> Any:
    """Normalize values for comparison (remove dynamic fields)."""
    if isinstance(value, dict):
        normalized = {}
        for k, v in value.items():
            # Skip dynamic fields
            if k in ["file_section_id", "chunk_id", "template_section_id", "anchor", "created_at", "timestamp"]:
                normalized[k] = "<DYNAMIC>" if isinstance(v, str) else v
            elif k == "confidence" or k == "relevancy_score" or k == "relevancy_score_if_unmapped":
                # Round confidence scores for comparison
                if isinstance(v, (int, float)):
                    normalized[k] = round(v, 2)
                else:
                    normalized[k] = v
            elif k == "template_path" and isinstance(v, list):
                # Keep template_path structure
                normalized[k] = v
            elif k == "sections" and isinstance(v, list):
                # Normalize sections but keep structure
                normalized[k] = [
                    normalize_for_comparison(section, f"{path}.sections[{i}]")
                    for i, section in enumerate(v)
                ]
            else:
                normalized[k] = normalize_for_comparison(v, f"{path}.{k}")
        return normalized
    elif isinstance(value, list):
        return [normalize_for_comparison(item, f"{path}[{i}]") for i, item in enumerate(value)]
    else:
        return value


def compare_results(actual: Dict[str, Any], expected: Dict[str, Any], node_name: str) -> tuple[bool, List[str]]:
    """Compare actual vs expected results and return (pass, reasons)."""
    reasons = []
    passed = True
    
    # Normalize both for comparison
    actual_norm = normalize_for_comparison(actual)
    expected_norm = normalize_for_comparison(expected)
    
    # Check required fields based on node
    required_fields = {
        "heading_mapping_llm": ["index"],
        "title_improver_llm": ["index"],
        "relevancy_scoring_llm": ["index"],
        "second_pass_mapping_llm": ["index"],
    }
    
    for field in required_fields.get(node_name, []):
        if field not in actual:
            passed = False
            reasons.append(f"✗ Missing required field: {field}")
        elif field not in expected:
            reasons.append(f"⚠ Expected result missing field: {field} (checking structure only)")
    
    # Check index.sections structure
    if "index" in actual_norm and "index" in expected_norm:
        actual_sections = actual_norm["index"].get("sections", [])
        expected_sections = expected_norm["index"].get("sections", [])
        
        if len(actual_sections) != len(expected_sections):
            passed = False
            reasons.append(f"✗ Section count mismatch: actual={len(actual_sections)}, expected={len(expected_sections)}")
        else:
            reasons.append(f"✓ Section count matches: {len(actual_sections)}")
        
        # Check section status distribution
        actual_mapped = sum(1 for s in actual_sections if s.get("status") == "mapped")
        expected_mapped = sum(1 for s in expected_sections if s.get("status") == "mapped")
        
        if node_name == "heading_mapping_llm":
            if abs(actual_mapped - expected_mapped) > 2:  # Allow some variance for LLM
                reasons.append(f"⚠ Mapped count variance: actual={actual_mapped}, expected={expected_mapped} (LLM variance acceptable)")
            else:
                reasons.append(f"✓ Mapped count reasonable: {actual_mapped} (expected ~{expected_mapped})")
        
        # Check that mapped sections have template_mapping
        for i, section in enumerate(actual_sections):
            if section.get("status") == "mapped":
                if not section.get("template_mapping", {}).get("mapped"):
                    passed = False
                    reasons.append(f"✗ Section {i} marked as mapped but template_mapping.mapped is not True")
                if not section.get("template_mapping", {}).get("template_section_id"):
                    reasons.append(f"⚠ Section {i} mapped but missing template_section_id")
                if "confidence" not in section.get("template_mapping", {}):
                    reasons.append(f"⚠ Section {i} mapped but missing confidence score")
    
    # Check index.stats
    if "index" in actual_norm and "index" in expected_norm:
        actual_stats = actual_norm["index"].get("stats", {})
        expected_stats = expected_norm["index"].get("stats", {})
        
        if actual_stats and expected_stats:
            for stat_key in ["total_sections", "mapped_sections", "unmapped_sections"]:
                if stat_key in actual_stats and stat_key in expected_stats:
                    reasons.append(f"✓ Stat '{stat_key}': actual={actual_stats[stat_key]}, expected={expected_stats[stat_key]}")
    
    return passed, reasons


def test_node_1_heading_mapping_llm() -> tuple[Dict[str, Any], bool, List[str]]:
    """Test heading_mapping_llm node."""
    print("\n" + "="*70)
    print("Test 1: heading_mapping_llm")
    print("="*70)
    
    try:
        state, index, template, chunks = load_phase2_results()
        agent = DocReviewAgent()
        
        print(f"  Input: {len(index.get('sections', []))} sections, template loaded")
        
        # Run the node
        updated_state = agent._node_heading_mapping_llm(state.copy())
        
        # Extract relevant output
        result = {
            "file_id": FILE_ID,
            "index": {
                "sections": updated_state.get("index", {}).get("sections", []),
                "stats": updated_state.get("index", {}).get("stats", {})
            }
        }
        
        save_result("heading_mapping_llm", result)
        
        # Load expected results
        if EXPECTED_FILE.exists():
            with EXPECTED_FILE.open("r", encoding="utf-8") as f:
                expected_data = json.load(f)
            expected = expected_data.get("heading_mapping_llm", {})
            passed, reasons = compare_results(result, expected, "heading_mapping_llm")
        else:
            passed = True
            reasons = ["⚠ No expected results file found - structure check only"]
        
        return result, passed, reasons
        
    except LLMNotAvailableError:
        print("  ⚠ LLM not available - skipping test")
        return {}, False, ["⚠ LLM not configured - cannot test Phase 3 nodes"]
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_node_2_title_improver_llm() -> tuple[Dict[str, Any], bool, List[str]]:
    """Test title_improver_llm node."""
    print("\n" + "="*70)
    print("Test 2: title_improver_llm")
    print("="*70)
    
    try:
        state, index, template, chunks = load_phase2_results()
        agent = DocReviewAgent()
        
        # First run heading_mapping_llm to get mapped sections
        state = agent._node_heading_mapping_llm(state.copy())
        
        mapped_count = sum(1 for s in state.get("index", {}).get("sections", []) if s.get("status") == "mapped")
        print(f"  Input: {mapped_count} mapped sections from previous node")
        
        # Run the node
        updated_state = agent._node_title_improver_llm(state.copy())
        
        # Extract relevant output
        result = {
            "file_id": FILE_ID,
            "index": {
                "sections": updated_state.get("index", {}).get("sections", [])
            }
        }
        
        save_result("title_improver_llm", result)
        
        # Load expected results
        if EXPECTED_FILE.exists():
            with EXPECTED_FILE.open("r", encoding="utf-8") as f:
                expected_data = json.load(f)
            expected = expected_data.get("title_improver_llm", {})
            passed, reasons = compare_results(result, expected, "title_improver_llm")
        else:
            passed = True
            reasons = ["⚠ No expected results file found - structure check only"]
        
        return result, passed, reasons
        
    except LLMNotAvailableError:
        print("  ⚠ LLM not available - skipping test")
        return {}, False, ["⚠ LLM not configured - cannot test Phase 3 nodes"]
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_node_3_relevancy_scoring_llm() -> tuple[Dict[str, Any], bool, List[str]]:
    """Test relevancy_scoring_llm node."""
    print("\n" + "="*70)
    print("Test 3: relevancy_scoring_llm")
    print("="*70)
    
    try:
        state, index, template, chunks = load_phase2_results()
        agent = DocReviewAgent()
        
        # First run heading_mapping_llm to get unmapped sections
        state = agent._node_heading_mapping_llm(state.copy())
        
        unmapped_count = sum(1 for s in state.get("index", {}).get("sections", []) if s.get("status") == "unmapped")
        print(f"  Input: {unmapped_count} unmapped sections from previous node")
        
        # Run the node
        updated_state = agent._node_relevancy_scoring_llm(state.copy())
        
        # Extract relevant output
        result = {
            "file_id": FILE_ID,
            "index": {
                "sections": updated_state.get("index", {}).get("sections", [])
            }
        }
        
        save_result("relevancy_scoring_llm", result)
        
        # Load expected results
        if EXPECTED_FILE.exists():
            with EXPECTED_FILE.open("r", encoding="utf-8") as f:
                expected_data = json.load(f)
            expected = expected_data.get("relevancy_scoring_llm", {})
            passed, reasons = compare_results(result, expected, "relevancy_scoring_llm")
        else:
            passed = True
            reasons = ["⚠ No expected results file found - structure check only"]
        
        return result, passed, reasons
        
    except LLMNotAvailableError:
        print("  ⚠ LLM not available - skipping test")
        return {}, False, ["⚠ LLM not configured - cannot test Phase 3 nodes"]
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_node_4_second_pass_mapping_llm() -> tuple[Dict[str, Any], bool, List[str]]:
    """Test second_pass_mapping_llm node."""
    print("\n" + "="*70)
    print("Test 4: second_pass_mapping_llm")
    print("="*70)
    
    try:
        state, index, template, chunks = load_phase2_results()
        agent = DocReviewAgent()
        
        # Run previous nodes to prepare state
        state = agent._node_heading_mapping_llm(state.copy())
        state = agent._node_relevancy_scoring_llm(state.copy())
        
        candidates_count = sum(
            1 for s in state.get("index", {}).get("sections", [])
            if s.get("status") == "unmapped"
            and (s.get("relevancy_score_if_unmapped") or 0) >= 0.4
        )
        print(f"  Input: {candidates_count} candidates above threshold (0.4)")
        
        # Run the node
        updated_state = agent._node_second_pass_mapping_llm(state.copy())
        
        # Extract relevant output
        result = {
            "file_id": FILE_ID,
            "index": {
                "sections": updated_state.get("index", {}).get("sections", []),
                "stats": updated_state.get("index", {}).get("stats", {})
            }
        }
        
        save_result("second_pass_mapping_llm", result)
        
        # Load expected results
        if EXPECTED_FILE.exists():
            with EXPECTED_FILE.open("r", encoding="utf-8") as f:
                expected_data = json.load(f)
            expected = expected_data.get("second_pass_mapping_llm", {})
            passed, reasons = compare_results(result, expected, "second_pass_mapping_llm")
        else:
            passed = True
            reasons = ["⚠ No expected results file found - structure check only"]
        
        return result, passed, reasons
        
    except LLMNotAvailableError:
        print("  ⚠ LLM not available - skipping test")
        return {}, False, ["⚠ LLM not configured - cannot test Phase 3 nodes"]
    except Exception as e:
        print(f"  ✗ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def main():
    """Run all Phase 3 tests."""
    print("\n" + "="*70)
    print("PHASE 3 TESTING - Template Mapping & Refinement (LLM-driven)")
    print("="*70)
    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Expected results: {EXPECTED_FILE}")
    print("\nNote: Phase 3 requires LLM to be configured and available.")
    print("      Results may vary due to LLM non-determinism.\n")
    
    ensure_dir(RESULTS_DIR)
    
    # Check if Phase 2 results exist
    if not PHASE2_INDEX_FILE.exists():
        print(f"✗ Phase 2 results not found: {PHASE2_INDEX_FILE}")
        print("  Please run Phase 2 tests first (test_phase2_tools.py)")
        return
    
    results = {}
    all_passed = True
    
    # Test each node
    result1, passed1, reasons1 = test_node_1_heading_mapping_llm()
    results["heading_mapping_llm"] = {"passed": passed1, "reasons": reasons1}
    if not passed1:
        all_passed = False
    
    result2, passed2, reasons2 = test_node_2_title_improver_llm()
    results["title_improver_llm"] = {"passed": passed2, "reasons": reasons2}
    if not passed2:
        all_passed = False
    
    result3, passed3, reasons3 = test_node_3_relevancy_scoring_llm()
    results["relevancy_scoring_llm"] = {"passed": passed3, "reasons": reasons3}
    if not passed3:
        all_passed = False
    
    result4, passed4, reasons4 = test_node_4_second_pass_mapping_llm()
    results["second_pass_mapping_llm"] = {"passed": passed4, "reasons": reasons4}
    if not passed4:
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 3 TEST SUMMARY")
    print("="*70)
    
    for node_name, result in results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"\n{node_name}: {status}")
        for reason in result["reasons"]:
            print(f"  {reason}")
    
    print("\n" + "="*70)
    if all_passed:
        print("OVERALL: ✓ ALL TESTS PASSED (with acceptable LLM variance)")
    else:
        print("OVERALL: ✗ SOME TESTS FAILED")
    print("="*70)
    
    # Save summary
    summary_file = RESULTS_DIR / "PHASE3_SUMMARY.md"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Phase 3 Test Results Summary\n\n")
        f.write(f"**Overall Status:** {'✓ PASS' if all_passed else '✗ FAIL'}\n\n")
        f.write("## Test Results\n\n")
        for node_name, result in results.items():
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            f.write(f"### {node_name}: {status}\n\n")
            for reason in result["reasons"]:
                f.write(f"- {reason}\n")
            f.write("\n")
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()

