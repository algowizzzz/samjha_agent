#!/usr/bin/env python3
"""Run Phase 1 → Phase 2 → Phase 3 pipeline without the UI."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.agent.doc_review_agent import DocReviewAgent


def pretty(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="data/docreview/Template_Version.pdf",
        help="Document path to process",
    )
    parser.add_argument(
        "--severity-filter",
        default="high",
        help="Apply only changes of this severity during Phase 3 (default: high)",
    )
    args = parser.parse_args()

    load_dotenv()

    doc_path = Path(args.path).expanduser().resolve()
    if not doc_path.exists():
        raise SystemExit(f"Document not found: {doc_path}")

    agent = DocReviewAgent()

    print(f"Running Phase 1 on {doc_path} ...")
    state = agent.run_phase1(str(doc_path))
    print(f"Phase 1 status: {state['phase1_status']}")

    print("\nRunning Phase 2 ...")
    state = agent.run_phase2(state)
    print(f"Phase 2 status: {state['phase2_status']}")
    print(f"Sections extracted: {len(state['phase2']['chunks'])}")
    print(f"Suggested changes: {len(state['changes']['suggested_changes'])}")

    print("\nRunning Phase 3 ...")
    state = agent.run_phase3(state, severity_filter=args.severity_filter)
    print(f"Phase 3 status: {state['phase3_status']}")
    print(
        pretty(
            {
                "applied_change_ids": state["changes"].get("applied_change_ids", []),
                "failed_changes": state["changes"].get("failed_changes", []),
            }
        )
    )


if __name__ == "__main__":
    main()

