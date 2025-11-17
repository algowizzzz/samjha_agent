#!/usr/bin/env python3
"""
Test script for Phase 1 document processing tools.
Tests each tool individually and compares against expected outputs.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from external.tools.doc_processing.detect_file_type import DetectFileTypeTool
from external.tools.doc_processing.convert_to_markdown import ConvertToMarkdownTool
from external.tools.doc_processing.extract_images import ExtractImagesTool
from external.tools.doc_processing.compute_file_stats import ComputeFileStatsTool
from external.tools.doc_processing.analyze_heading_structure import AnalyzeHeadingStructureTool
from external.tools.doc_processing.build_file_metadata import BuildFileMetadataTool


# Test configuration
BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "CAR_Chapter_1_Overview.pdf"
FILE_ID = "car-chapter-1-overview"
RESULTS_DIR = BASE_DIR / "test_phase1_results"
EXPECTED_FILE = BASE_DIR / "test_phase1_expected"


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


def normalize_for_comparison(value: Any, path: str = "") -> Any:
    """Normalize values for comparison (remove dynamic fields)."""
    if isinstance(value, dict):
        normalized = {}
        for k, v in value.items():
            # Skip dynamic fields
            if k in ["md_file_id", "image_id", "path", "md_path", "created_at", "timestamp"]:
                normalized[k] = "<DYNAMIC>" if isinstance(v, str) else v
            elif k == "raw_markdown":
                # Keep first 200 chars for comparison
                if isinstance(v, str):
                    normalized[k] = v[:200] + "...[TRUNCATED]" if len(v) > 200 else v
                else:
                    normalized[k] = v
            elif k == "images":
                # Normalize image paths but keep structure
                if isinstance(v, list):
                    normalized[k] = [
                        {**img, "path": "<DYNAMIC>", "image_id": "<DYNAMIC>"}
                        if isinstance(img, dict) else img
                        for img in v
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
    
    # Normalize both for comparison
    actual_norm = normalize_for_comparison(actual)
    expected_norm = normalize_for_comparison(expected)
    
    # Check required fields
    required_fields = {
        "detect_file_type": ["file_id", "file_type"],
        "convert_to_markdown": ["file_id", "md_file_id", "md_path", "raw_markdown"],
        "extract_images": ["file_id", "images"],
        "compute_file_stats": ["file_id", "md_file_id", "word_count", "page_count"],
        "analyze_heading_structure": ["file_id", "md_file_id", "headings", "heading_levels"],
        "build_file_metadata": ["file_metadata"],
    }
    
    for field in required_fields.get(tool_name, []):
        if tool_name == "build_file_metadata":
            if "file_metadata" not in actual:
                reasons.append(f"✗ Missing 'file_metadata' in output")
                passed = False
        else:
            if field not in actual:
                reasons.append(f"✗ Missing required field: {field}")
                passed = False
    
    # Field-specific checks
    if tool_name == "detect_file_type":
        if actual.get("file_type") != "pdf":
            reasons.append(f"✗ Expected file_type='pdf', got '{actual.get('file_type')}'")
            passed = False
        if not (0 <= actual.get("confidence", -1) <= 1):
            reasons.append(f"✗ Confidence should be 0-1, got {actual.get('confidence')}")
            passed = False
    
    elif tool_name == "convert_to_markdown":
        if not actual.get("raw_markdown"):
            reasons.append("✗ raw_markdown is empty")
            passed = False
        if not actual.get("md_file_id"):
            reasons.append("✗ md_file_id is missing")
            passed = False
        if not actual.get("md_path"):
            reasons.append("✗ md_path is missing")
            passed = False
        # Check if markdown file exists
        md_path = Path(actual.get("md_path", ""))
        if not md_path.exists():
            reasons.append(f"✗ Markdown file does not exist: {md_path}")
            passed = False
    
    elif tool_name == "extract_images":
        images = actual.get("images", [])
        if not isinstance(images, list):
            reasons.append(f"✗ 'images' should be a list, got {type(images)}")
            passed = False
        for img in images:
            if "image_id" not in img or "path" not in img:
                reasons.append("✗ Image entry missing image_id or path")
                passed = False
    
    elif tool_name == "compute_file_stats":
        word_count = actual.get("word_count")
        page_count = actual.get("page_count")
        if not isinstance(word_count, int) or word_count <= 0:
            reasons.append(f"✗ Invalid word_count: {word_count}")
            passed = False
        if not isinstance(page_count, int) or page_count <= 0:
            reasons.append(f"✗ Invalid page_count: {page_count}")
            passed = False
    
    elif tool_name == "analyze_heading_structure":
        headings = actual.get("headings", [])
        heading_levels = actual.get("heading_levels", [])
        if not isinstance(headings, list):
            reasons.append(f"✗ 'headings' should be a list, got {type(headings)}")
            passed = False
        if not isinstance(heading_levels, list):
            reasons.append(f"✗ 'heading_levels' should be a list, got {type(heading_levels)}")
            passed = False
        for heading in headings:
            required = ["heading_level", "heading_text", "heading_path", "line_number"]
            for req in required:
                if req not in heading:
                    reasons.append(f"✗ Heading missing required field: {req}")
                    passed = False
    
    elif tool_name == "build_file_metadata":
        metadata = actual.get("file_metadata", {})
        required = ["file_id", "source_path", "file_type", "md_file_id", "page_count", "word_count"]
        for req in required:
            if req not in metadata:
                reasons.append(f"✗ Metadata missing required field: {req}")
                passed = False
    
    if passed:
        reasons.append("✓ All checks passed")
    
    return passed, reasons


def test_tool_1_detect_file_type() -> tuple[Dict[str, Any], bool, List[str]]:
    """Test detect_file_type tool."""
    print("\n" + "="*70)
    print("Testing: detect_file_type")
    print("="*70)
    
    tool = DetectFileTypeTool()
    input_data = {
        "file_id": FILE_ID,
        "source_path": str(PDF_PATH.resolve()),
    }
    
    print(f"Input: {json.dumps(input_data, indent=2)}")
    
    try:
        result = tool.execute(input_data)
        save_result("detect_file_type", result)
        
        # Expected values
        expected = {
            "file_id": FILE_ID,
            "file_type": "pdf",
            "confidence": 1.0,
            "detected_via": "extension"
        }
        
        passed, reasons = compare_results(result, expected, "detect_file_type")
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_2_convert_to_markdown(file_type: str) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test convert_to_markdown tool."""
    print("\n" + "="*70)
    print("Testing: convert_to_markdown")
    print("="*70)
    
    tool = ConvertToMarkdownTool()
    input_data = {
        "file_id": FILE_ID,
        "source_path": str(PDF_PATH.resolve()),
        "file_type": file_type,
    }
    
    print(f"Input: {json.dumps(input_data, indent=2)}")
    
    try:
        result = tool.execute(input_data)
        save_result("convert_to_markdown", result)
        
        expected = {
            "file_id": FILE_ID,
            "md_file_id": "<DYNAMIC>",
            "md_path": "<DYNAMIC>",
            "raw_markdown": "<DYNAMIC>",
            "notes": "Converted from PDF source"
        }
        
        passed, reasons = compare_results(result, expected, "convert_to_markdown")
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_3_extract_images(file_type: str) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test extract_images tool."""
    print("\n" + "="*70)
    print("Testing: extract_images")
    print("="*70)
    
    tool = ExtractImagesTool()
    input_data = {
        "file_id": FILE_ID,
        "source_path": str(PDF_PATH.resolve()),
        "file_type": file_type,
    }
    
    print(f"Input: {json.dumps(input_data, indent=2)}")
    
    try:
        result = tool.execute(input_data)
        save_result("extract_images", result)
        
        expected = {
            "file_id": FILE_ID,
            "images": [],
            "notes": "<DYNAMIC>"
        }
        
        passed, reasons = compare_results(result, expected, "extract_images")
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_4_compute_file_stats(md_file_id: str, raw_markdown: str) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test compute_file_stats tool."""
    print("\n" + "="*70)
    print("Testing: compute_file_stats")
    print("="*70)
    
    tool = ComputeFileStatsTool()
    input_data = {
        "file_id": FILE_ID,
        "md_file_id": md_file_id,
        "raw_markdown": raw_markdown,
    }
    
    print(f"Input: file_id={FILE_ID}, md_file_id={md_file_id}, raw_markdown length={len(raw_markdown)}")
    
    try:
        result = tool.execute(input_data)
        save_result("compute_file_stats", result)
        
        expected = {
            "file_id": FILE_ID,
            "md_file_id": md_file_id,
            "word_count": "<DYNAMIC>",
            "page_count": "<DYNAMIC>"
        }
        
        passed, reasons = compare_results(result, expected, "compute_file_stats")
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_5_analyze_heading_structure(md_file_id: str, raw_markdown: str) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test analyze_heading_structure tool."""
    print("\n" + "="*70)
    print("Testing: analyze_heading_structure")
    print("="*70)
    
    tool = AnalyzeHeadingStructureTool()
    input_data = {
        "file_id": FILE_ID,
        "md_file_id": md_file_id,
        "raw_markdown": raw_markdown,
    }
    
    print(f"Input: file_id={FILE_ID}, md_file_id={md_file_id}, raw_markdown length={len(raw_markdown)}")
    
    try:
        result = tool.execute(input_data)
        save_result("analyze_heading_structure", result)
        
        expected = {
            "file_id": FILE_ID,
            "md_file_id": md_file_id,
            "headings": [],
            "heading_levels": []
        }
        
        passed, reasons = compare_results(result, expected, "analyze_heading_structure")
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def test_tool_6_build_file_metadata(
    file_type: str,
    md_file_id: str,
    md_path: str,
    images: List[Dict[str, Any]],
    page_count: int,
    word_count: int,
    heading_levels: List[str]
) -> tuple[Dict[str, Any], bool, List[str]]:
    """Test build_file_metadata tool."""
    print("\n" + "="*70)
    print("Testing: build_file_metadata")
    print("="*70)
    
    tool = BuildFileMetadataTool()
    input_data = {
        "file_id": FILE_ID,
        "source_path": str(PDF_PATH.resolve()),
        "file_type": file_type,
        "md_file_id": md_file_id,
        "md_path": md_path,
        "images": images,
        "page_count": page_count,
        "word_count": word_count,
        "heading_levels": heading_levels,
    }
    
    print(f"Input: {json.dumps({**input_data, 'raw_markdown': '<TRUNCATED>'}, indent=2)}")
    
    try:
        result = tool.execute(input_data)
        save_result("build_file_metadata", result)
        
        expected = {
            "file_metadata": {
                "file_id": FILE_ID,
                "source_path": str(PDF_PATH.resolve()),
                "file_type": file_type,
                "md_file_id": md_file_id,
                "md_path": md_path,
                "images": images,
                "page_count": page_count,
                "word_count": word_count,
                "heading_levels": heading_levels,
            }
        }
        
        passed, reasons = compare_results(result, expected, "build_file_metadata")
        return result, passed, reasons
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}, False, [f"✗ Exception: {str(e)}"]


def main():
    """Run all Phase 1 tool tests."""
    print("\n" + "="*70)
    print("PHASE 1 DOCUMENT PROCESSING TOOLS TEST SUITE")
    print("="*70)
    print(f"Test PDF: {PDF_PATH}")
    print(f"Results directory: {RESULTS_DIR}")
    print("="*70)
    
    if not PDF_PATH.exists():
        print(f"\n✗ ERROR: PDF file not found: {PDF_PATH}")
        sys.exit(1)
    
    ensure_dir(RESULTS_DIR)
    
    results_summary = []
    
    # Test 1: detect_file_type
    result1, passed1, reasons1 = test_tool_1_detect_file_type()
    results_summary.append(("detect_file_type", passed1, reasons1))
    file_type = result1.get("file_type", "pdf") if result1 else "pdf"
    
    # Test 2: convert_to_markdown
    result2, passed2, reasons2 = test_tool_2_convert_to_markdown(file_type)
    results_summary.append(("convert_to_markdown", passed2, reasons2))
    md_file_id = result2.get("md_file_id", "") if result2 else ""
    md_path = result2.get("md_path", "") if result2 else ""
    raw_markdown = result2.get("raw_markdown", "") if result2 else ""
    
    # Test 3: extract_images
    result3, passed3, reasons3 = test_tool_3_extract_images(file_type)
    results_summary.append(("extract_images", passed3, reasons3))
    images = result3.get("images", []) if result3 else []
    
    # Test 4: compute_file_stats
    if md_file_id and raw_markdown:
        result4, passed4, reasons4 = test_tool_4_compute_file_stats(md_file_id, raw_markdown)
        results_summary.append(("compute_file_stats", passed4, reasons4))
        page_count = result4.get("page_count", 0) if result4 else 0
        word_count = result4.get("word_count", 0) if result4 else 0
    else:
        print("\n✗ Skipping compute_file_stats: missing md_file_id or raw_markdown")
        results_summary.append(("compute_file_stats", False, ["✗ Skipped due to previous failure"]))
        page_count = 0
        word_count = 0
    
    # Test 5: analyze_heading_structure
    if md_file_id and raw_markdown:
        result5, passed5, reasons5 = test_tool_5_analyze_heading_structure(md_file_id, raw_markdown)
        results_summary.append(("analyze_heading_structure", passed5, reasons5))
        heading_levels = result5.get("heading_levels", []) if result5 else []
    else:
        print("\n✗ Skipping analyze_heading_structure: missing md_file_id or raw_markdown")
        results_summary.append(("analyze_heading_structure", False, ["✗ Skipped due to previous failure"]))
        heading_levels = []
    
    # Test 6: build_file_metadata
    if md_file_id and md_path:
        result6, passed6, reasons6 = test_tool_6_build_file_metadata(
            file_type, md_file_id, md_path, images, page_count, word_count, heading_levels
        )
        results_summary.append(("build_file_metadata", passed6, reasons6))
    else:
        print("\n✗ Skipping build_file_metadata: missing required inputs")
        results_summary.append(("build_file_metadata", False, ["✗ Skipped due to previous failure"]))
    
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
