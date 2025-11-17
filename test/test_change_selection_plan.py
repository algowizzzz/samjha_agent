import unittest
from pathlib import Path
from unittest.mock import patch

from external.agent.doc_review_agent import DocReviewAgent


class ChangeSelectionPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DocReviewAgent()
        self.state = self.agent._initialise_state(
            source=Path(__file__),
            run_id="testrun-change-plan",
            template_id="policy_template",
        )
        self.state["structure"]["raw_text"] = "Original content before changes."
        self.state["changes"]["suggested_changes"] = [
            {
                "id": "CHG-001",
                "index": 1,
                "section_title": "Overview",
                "type": "clarity",
                "severity": "medium",
                "original_text": "Replace this overview sentence.",
                "suggested_text": "Improved overview sentence.",
                "status": "pending",
            },
            {
                "id": "CHG-002",
                "index": 2,
                "section_title": "Scope",
                "type": "structural",
                "severity": "high",
                "original_text": "Old scope paragraph.",
                "suggested_text": "Updated scope paragraph.",
                "status": "pending",
            },
            {
                "id": "CHG-003",
                "index": 3,
                "section_title": "Escalations",
                "type": "missing_content",
                "severity": "high",
                "original_text": "",
                "suggested_text": "Add escalation matrix.",
                "status": "pending",
            },
        ]

    def test_select_changes_respects_plan_ids(self) -> None:
        plan = {
            "apply_mode": "by_ids",
            "change_ids_to_apply": ["CHG-002", "CHG-003"],
            "rationale": "Apply requested ids",
        }

        selected = self.agent._select_changes_for_application(self.state, plan=plan)
        selected_ids = [change["id"] for change in selected]

        # CHG-003 should be skipped because it lacks original_text, even if requested.
        self.assertEqual(selected_ids, ["CHG-002"])
        self.assertEqual(len(self.state["changes"]["skipped_changes"]), 1)
        self.assertEqual(
            self.state["changes"]["skipped_changes"][0]["change"]["id"],
            "CHG-003",
        )

    def test_interpret_change_instruction_filters_invalid_ids(self) -> None:
        with patch.object(
            DocReviewAgent,
            "_invoke_llm_prompt",
            return_value={
                "apply_mode": "by_ids",
                "change_ids_to_apply": ["CHG-999", "CHG-002"],
                "rationale": "User said apply high severity",
            },
        ):
            plan = self.agent.interpret_change_instruction(self.state, "Apply high severity")

        self.assertIsNotNone(plan)
        self.assertEqual(plan["change_ids_to_apply"], ["CHG-002"])
        self.assertEqual(
            self.state["changes"]["change_selection_plan"]["change_ids_to_apply"],
            ["CHG-002"],
        )


if __name__ == "__main__":
    unittest.main()

