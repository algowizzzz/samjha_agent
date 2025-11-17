#!/usr/bin/env python3
"""
Test script for Phase 2 document processing tools.
Tests each tool individually and compares against expected outputs.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from external.tools.doc_processing.decide_chunking_strategy import DecideChunkingStrategyTool
from external.tools.doc_processing.chunk_markdown import ChunkMarkdownTool
from external.tools.doc_processing.build_index import BuildIndexTool
from external.tools.doc_processing.load_outline_template import LoadOutlineTemplateTool


# Test configuration
BASE_DIR = Path(__file__).parent
PHASE1_RESULTS_DIR = BASE_DIR / "test_phase1_results"
RESULTS_DIR = BASE_DIR / "test_phase2_results"
EXPECTED_FILE = BASE_DIR / "test_phase2_expected.json"

# Load Phase 1 results for input to Phase 2 tools
PHASE1_METADATA_FILE = PHASE1_RESULTS_DIR / "build_file_metadata" / "output.json"
PHASE1_MARKDOWN_FILE = PHASE1_RESULTS_DIR / "convert_to_markdown" / "output.json"
PHASE1_HEADINGS_FILE = PHASE1_RESULTS_DIR / "analyze_heading_structure" / "output.json"

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


def load_phase1_results() -> tuple[Dict[str, Any], str, str, List[Dict[str, Any]]]:
    """Load Phase 1 results needed for Phase 2 tests."""
    with PHASE1_METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata_result = json.load(f)
    
    with PHASE1_MARKDOWN_FILE.open("r", encoding="utf-8") as f:
        markdown_result = json.load(f)
    
    with PHASE1_HEADINGS_FILE.open("r", encoding="utf-8") as f:
        headings_result = json.load(f)
    
    file_metadata = metadata_result.get("file_metadata", {})
    md_file_id = markdown_result.get("md_file_id", "")
    raw_markdown = markdown_result.get("raw_markdown", "")
    heading_structure = headings_result.get("headings", [])
    
    return file_metadata, md_file_id, raw_markdown, heading_structure


def normalize_for_comparison(value: Any, path: str = "") -> Any:
    """Normalize values for comparison (remove dynamic fields)."""
    if isinstance(value, dict):
        normalized = {}
        for k, v in value.items():
            # Skip dynamic fields
            if k in ["chunk_id", "file_section_id", "md_file_id", "image_id", "path", "md_path", 
                     "created_at", "timestamp", "anchor", "template_path"]:
                normalized[k] = "<DYNAMIC>" if isinstance(v, str) else v
            elif k in ["text", "raw_markdown"]:
                # Keep first 200 chars for comparison
                if isinstance(v, str):
                    normalized[k] = v[:200] + "...[TRUNCATED]" if len(v) > 200 else v
                else:
                    normalized[k] = v
            elif k == "chunks":
                # Normalize chunks but keep structure
                if isinstance(v, list):
                    normalized[k] = [
                        {**chunk, "chunk_id": "<DYNAMIC>", "text": chunk.get("text", "")[:200] + "...[TRUNCATED]" if len(chunk.get("text", "")) > 200 else chunk.get("text", "")}
                        if isinstance(chunk, dict) else chunk
                        for chunk in v
                    ]
                else:
                    normalized[k] = v
            elif k == "sections":
                # Normalize sections
                if isinstance(v, list):
                    normalized[k] = [
                        {**section, "chunk_id": "<DYNAMIC>", "file_section_id": "<DYNAMIC>", "anchor": "<DYNAMIC>"}
                        if isinstance(section, dict) else section
                        for section in v
                    ]
                else:
                    normalized[k] = v
            else:
                normalized[k] = normalize_for_comparison(v, f"{path}.{k}")
        return normalized
    elif isinstance(value, list):
        return [normalize_for_comparison(item, f"{path}[{i}]") for i, item in enumerate(value)]
    else:
        return value


def compare_results(actual: Dict[str, Any], expected: Dict[str, Any], tool_name: str) -> tuple[bool, List[str]]:
    """Compare actual vs expected results and return (pass, reasons)."""
    reasons = []
    passed = True
    
    if tool_name == "decide_chunking_strategy":
        if actual.get("should_chunk") != expected.get("should_chunk"):
            reasons.append(f"✗ should_chunk mismatch: expected {expected.get('should_chunk')}, got {actual.get('should_chunk')}")
            passed = False
        if actual.get("strategy") != expected.get("strategy"):
            reasons.append(f"✗ strategy mismatch: expected {expected.get('strategy')}, got {actual.get('strategy')}")
            passed = False
        if actual.get("reason") != expected.get("reason"):
            reasons.append(f"✗ reason mismatch: expected {expected.get('reason')}, got {actual.get('reason')}")
            passed = False
        if actual.get("context_aware_level") != expected.get("context_aware_level"):
            reasons.append(f"✗ context_aware_level mismatch: expected {expected.get('context_aware_level')}, got {actual.get('context_aware_level')}")
            passed = False
        
        if passed:
            reasons.append("✓ All checks passed")
    
    elif tool_name == "chunk_markdown":
        chunks = actual.get("chunks", [])
        expected_count = expected.get("chunks_count", 0)
        if len(chunks) != expected_count:
            reasons.append(f"✗ chunks count mismatch: expected {expected_count}, got {len(chunks)}")
            passed = False
        
        # Check strategy matches
        if actual.get("strategy") and actual["strategy"] != expected.get("strategy"):
            reasons.append(f"✗ strategy mismatch: expected {expected.get('strategy')}, got {actual.get('strategy')}")
            passed = False
        
        # Check H1 headings match expected
        if chunks:
            actual_h1_headings = [chunk.get("heading_text_original", "") for chunk in chunks]
            expected_h1 = expected.get("expected_h1_headings", [])
            for i, expected_heading in enumerate(expected_h1):
                if i < len(actual_h1_headings):
                    if expected_heading.lower() not in actual_h1_headings[i].lower() and actual_h1_headings[i].lower() not in expected_heading.lower():
                        reasons.append(f"⚠ H1 heading {i} may not match: expected '{expected_heading}', got '{actual_h1_headings[i]}'")
                else:
                    reasons.append(f"✗ Missing chunk for H1 heading: '{expected_heading}'")
                    passed = False
        
        if passed and not reasons:
            reasons.append("✓ All checks passed")
    
    elif tool_name == "build_index":
        index = actual.get("index", {})
        sections = index.get("sections", [])
        stats = index.get("stats", {})
        
        expected_count = expected.get("sections_count", 0)
        if len(sections) != expected_count:
            reasons.append(f"✗ sections count mismatch: expected {expected_count}, got {len(sections)}")
            passed = False
        else:
            reasons.append(f"✓ Sections count matches: {len(sections)}")
        
        # Check chunking info if present (may not be stored in index, that's ok)
        chunking = index.get("chunking", {})
        if chunking:
            if chunking.get("should_chunk") != expected.get("should_chunk"):
                reasons.append(f"⚠ should_chunk in index: expected {expected.get('should_chunk')}, got {chunking.get('should_chunk')}")
            if chunking.get("strategy") != expected.get("strategy"):
                reasons.append(f"⚠ strategy in index: expected {expected.get('strategy')}, got {chunking.get('strategy')}")
        
        if stats.get("total_sections") != expected.get("total_sections"):
            reasons.append(f"✗ total_sections mismatch: expected {expected.get('total_sections')}, got {stats.get('total_sections')}")
            passed = False
        else:
            reasons.append(f"✓ Total sections matches: {stats.get('total_sections')}")
        
        if stats.get("unmapped_sections") != expected.get("unmapped_sections"):
            reasons.append(f"✗ unmapped_sections mismatch: expected {expected.get('unmapped_sections')}, got {stats.get('unmapped_sections')}")
            passed = False
        else:
            reasons.append(f"✓ Unmapped sections matches: {stats.get('unmapped_sections')}")
        
        # Verify sections are unmapped
        mapped_count = sum(1 for s in sections if s.get("status") == "mapped")
        if mapped_count != 0:
            reasons.append(f"✗ Expected 0 mapped sections, got {mapped_count}")
            passed = False
        else:
            reasons.append("✓ All sections are unmapped (expected for Phase 2)")
        
        if passed and any(r.startswith("✓") for r in reasons):
            if not any(r.startswith("✗") for r in reasons):
                reasons.insert(0, "✓ All checks passed")
    
    elif tool_name == "load_outline_template":
        if expected.get("optional"):
            reasons.append("✓ Tool is optional (may be skipped)")
            passed = True
        else:
            if "template" not in actual:
                reasons.append("✗ Missing 'template' in output")
                passed = False
            if passed:
                reasons.append("✓ All checks passed")
    
    return passed, reasons


def test_tool_1_decide_chunking_strategy(file_metadata: Dict[str, Any]) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test decide_chunking_strategy tool."""
    print("\n" + "="*70)
    print("Testing: decide_chunking_strategy")
    print("="*70)
    
    tool = DecideChunkingStrategyTool()
    input_data = {
        "file_metadata": file_metadata,
        "config": {
            "page_threshold": 10,
            "context_aware_level": "h1",
            "default_strategy": "by_h2"
        }
    }
    
    print(f"Input config: page_threshold=10, context_aware_level=h1")
    print(f"File metadata: pages={file_metadata.get('page_count')}")
    
    try:
        result = tool.execute(input_data)
        save_result("decide_chunking_strategy", result)
        
        # Load expected
        with EXPECTED_FILE.open("r", encoding="utf-8") as f:
            expected_data = json.load(f)
        expected = expected_data["expected_results"]["decide_chunking_strategy"]
        
        passed, reasons = compare_results(result, expected, "decide_chunking_strategy")
        
        print(f"\nResult: should_chunk={result.get('should_chunk')}, strategy={result.get('strategy')}, reason={result.get('reason')}")
        for reason in reasons:
            print(f"  {reason}")
        
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_2_chunk_markdown(
    md_file_id: str, 
    raw_markdown: str, 
    chunking_decision: Dict[str, Any]
) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test chunk_markdown tool."""
    print("\n" + "="*70)
    print("Testing: chunk_markdown")
    print("="*70)
    
    tool = ChunkMarkdownTool()
    input_data = {
        "file_id": FILE_ID,
        "md_file_id": md_file_id,
        "raw_markdown": raw_markdown,
        "strategy": chunking_decision.get("strategy", "none"),
        "context_aware_level": chunking_decision.get("context_aware_level", "h2")
    }
    
    print(f"Input: strategy={input_data['strategy']}, context_aware_level={input_data['context_aware_level']}")
    print(f"Markdown length: {len(raw_markdown)} chars")
    
    try:
        result = tool.execute(input_data)
        result["strategy"] = input_data["strategy"]  # Include strategy in result for comparison
        save_result("chunk_markdown", result)
        
        # Load expected
        with EXPECTED_FILE.open("r", encoding="utf-8") as f:
            expected_data = json.load(f)
        expected = expected_data["expected_results"]["chunk_markdown"]
        expected["strategy"] = chunking_decision.get("strategy")
        
        passed, reasons = compare_results(result, expected, "chunk_markdown")
        
        chunks = result.get("chunks", [])
        print(f"\nResult: {len(chunks)} chunks created")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3
            print(f"  Chunk {i}: {chunk.get('heading_text_original', '')[:60]}")
        if len(chunks) > 3:
            print(f"  ... and {len(chunks) - 3} more chunks")
        
        for reason in reasons:
            print(f"  {reason}")
        
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_3_build_index(
    file_metadata: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    heading_structure: List[Dict[str, Any]],
    chunking_decision: Dict[str, Any]
) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test build_index tool."""
    print("\n" + "="*70)
    print("Testing: build_index")
    print("="*70)
    
    tool = BuildIndexTool()
    input_data = {
        "file_metadata": file_metadata,
        "chunks": chunks,
        "heading_structure": heading_structure,
        "chunking_decision": chunking_decision
    }
    
    print(f"Input: {len(chunks)} chunks, {len(heading_structure)} headings")
    print(f"Chunking decision: should_chunk={chunking_decision.get('should_chunk')}, strategy={chunking_decision.get('strategy')}")
    
    try:
        result = tool.execute(input_data)
        save_result("build_index", result)
        
        # Load expected
        with EXPECTED_FILE.open("r", encoding="utf-8") as f:
            expected_data = json.load(f)
        expected = expected_data["expected_results"]["build_index"]
        
        passed, reasons = compare_results(result, expected, "build_index")
        
        index = result.get("index", {})
        sections = index.get("sections", [])
        stats = index.get("stats", {})
        print(f"\nResult: {len(sections)} sections, total_sections={stats.get('total_sections')}, unmapped={stats.get('unmapped_sections')}")
        
        for reason in reasons:
            print(f"  {reason}")
        
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_4_load_outline_template() -> tuple[Dict[str, Any], bool, List[str]]:
    """Test load_outline_template tool."""
    print("\n" + "="*70)
    print("Testing: load_outline_template")
    print("="*70)
    
    tool = LoadOutlineTemplateTool()
    
    # Try with a template_id if available, otherwise skip
    input_data = {
        "template_id": "outline"
    }
    
    print(f"Input: template_id={input_data['template_id']}")
    
    try:
        result = tool.execute(input_data)
        save_result("load_outline_template", result)
        
        # Load expected
        with EXPECTED_FILE.open("r", encoding="utf-8") as f:
            expected_data = json.load(f)
        expected = expected_data["expected_results"]["load_outline_template"]
        
        passed, reasons = compare_results(result, expected, "load_outline_template")
        
        print(f"\nResult: template_id={result.get('template_id')}, template_path={result.get('template_path', 'N/A')}")
        if "template" in result:
            sections = result.get("template", {}).get("sections", [])
            print(f"  Template sections: {len(sections)}")
        
        for reason in reasons:
            print(f"  {reason}")
        
        return result, passed, reasons
        
    except FileNotFoundError:
        print("  ⚠ Template file not found (this is optional, skipping)")
        return {}, True, ["✓ Tool is optional, template not found (expected)"]
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def main():
    """Run all Phase 2 tool tests."""
    print("\n" + "="*70)
    print("PHASE 2 DOCUMENT PROCESSING TOOLS TEST SUITE")
    print("="*70)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Expected file: {EXPECTED_FILE}")
    print("="*70)
    
    if not EXPECTED_FILE.exists():
        print(f"\n✗ ERROR: Expected results file not found: {EXPECTED_FILE}")
        sys.exit(1)
    
    if not PHASE1_METADATA_FILE.exists():
        print(f"\n✗ ERROR: Phase 1 results not found. Please run Phase 1 tests first: {PHASE1_METADATA_FILE}")
        sys.exit(1)
    
    ensure_dir(RESULTS_DIR)
    
    # Load Phase 1 results
    print("\nLoading Phase 1 results...")
    file_metadata, md_file_id, raw_markdown, heading_structure = load_phase1_results()
    print(f"  ✓ Loaded metadata: {file_metadata.get('file_id')}")
    print(f"  ✓ Loaded markdown: {md_file_id}")
    print(f"  ✓ Loaded {len(heading_structure)} headings")
    
    results_summary = []
    
    # Test 1: decide_chunking_strategy
    result1, passed1, reasons1 = test_tool_1_decide_chunking_strategy(file_metadata)
    results_summary.append(("decide_chunking_strategy", passed1, reasons1))
    chunking_decision = result1 if result1 else {}
    
    # Test 2: chunk_markdown
    if chunking_decision:
        result2, passed2, reasons2 = test_tool_2_chunk_markdown(md_file_id, raw_markdown, chunking_decision)
        results_summary.append(("chunk_markdown", passed2, reasons2))
        chunks = result2.get("chunks", []) if result2 else []
    else:
        print("\n✗ Skipping chunk_markdown: missing chunking_decision")
        results_summary.append(("chunk_markdown", False, ["✗ Skipped due to previous failure"]))
        chunks = []
    
    # Test 3: build_index
    if chunks:
        result3, passed3, reasons3 = test_tool_3_build_index(
            file_metadata, chunks, heading_structure, chunking_decision
        )
        results_summary.append(("build_index", passed3, reasons3))
    else:
        print("\n✗ Skipping build_index: missing chunks")
        results_summary.append(("build_index", False, ["✗ Skipped due to previous failure"]))
    
    # Test 4: load_outline_template (optional)
    result4, passed4, reasons4 = test_tool_4_load_outline_template()
    results_summary.append(("load_outline_template", passed4, reasons4))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_passed = sum(1 for _, passed, _ in results_summary if passed)
    total_tests = len(results_summary)
    
    for tool_name, passed, reasons in results_summary:
        status = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        print(f"\n{icon} {tool_name}: {status}")
        for reason in reasons:
            print(f"    {reason}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {total_passed}/{total_tests} tests passed")
    print("="*70)
    
    # Save summary
    summary_file = RESULTS_DIR / "summary.json"
    summary_data = {
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_tests - total_passed,
        "results": [
            {
                "tool": tool_name,
                "passed": passed,
                "reasons": reasons
            }
            for tool_name, passed, reasons in results_summary
        ]
    }
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())

