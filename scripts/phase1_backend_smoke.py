#!/usr/bin/env python3
"""Run Phase 1 (ingestion + LLM reports) without the UI."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.agent.doc_review_agent import DocReviewAgent


def pretty(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="data/docreview/Template_Version.pdf",
        help="Path to the document to process",
    )
    parser.add_argument(
        "--template-id",
        help="Optional template override (defaults to policy_template)",
    )
    args = parser.parse_args()

    load_dotenv()

    agent = DocReviewAgent()
    doc_path = Path(args.path).expanduser().resolve()
    if not doc_path.exists():
        raise SystemExit(f"Document not found: {doc_path}")

    print(f"Running Phase 1 for {doc_path} ...")
    state = agent.run_phase1(str(doc_path), template_id=args.template_id)

    doc_meta = state["doc_meta"]
    print("\n=== Document Metadata ===")
    print(
        pretty(
            {
                "run_id": state["run_id"],
                "doc_title": doc_meta["doc_title"],
                "file_type": doc_meta.get("file_type"),
                "page_count": doc_meta.get("page_count"),
                "word_count": doc_meta.get("word_count"),
            }
        )
    )

    print("\n=== Phase 1 Stats ===")
    print(pretty(state["phase1"]["stats"]))
    print(f"Phase 1 status: {state['phase1_status']}")

    if state["phase1"].get("doc_summary"):
        print("\n=== Executive Summary ===")
        print(pretty(state["phase1"]["doc_summary"]))
    else:
        print("\n(No executive summary generated — check logs/LLM configuration.)")

    if state["phase1"].get("toc_review"):
        print("\n=== TOC & Structure Review ===")
        print(pretty(state["phase1"]["toc_review"]))

    if state["phase1"].get("template_fitness_report"):
        print("\n=== Template Fitness Deep Dive ===")
        print(
            pretty(
                {
                    "overall_alignment": state["phase1"]["template_fitness_report"].get(
                        "overall_alignment"
                    ),
                    "categories": state["phase1"]["template_fitness_report"].get("categories", []),
                }
            )
        )

    if state["phase1"].get("section_strategy"):
        print("\n=== Section Strategy ===")
        print(pretty(state["phase1"]["section_strategy"]))


if __name__ == "__main__":
    main()
