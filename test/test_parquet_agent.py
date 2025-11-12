import os
import pytest

from external.agent.parquet_agent import ParquetQueryAgent


class TestParquetAgent:
    def test_agent_minimal_run_produces_dual_outputs(self):
        agent = ParquetQueryAgent()
        state = agent.run_query(query="Show a small preview of data", session_id=None, user_id="tester1")

        assert isinstance(state, dict)
        assert state.get("control") == "end"
        final_output = state.get("final_output") or {}
        assert "response" in final_output
        assert "prompt_monitor" in final_output
        # Response should be a non-empty string
        assert isinstance(final_output["response"], str)
        assert isinstance(final_output["prompt_monitor"], str)

    def test_previous_sessions_populated_on_second_run(self):
        agent = ParquetQueryAgent()
        # First run
        _ = agent.run_query(query="First run", session_id=None, user_id="tester2")
        # Second run should include last sessions
        state = agent.run_query(query="Second run", session_id=None, user_id="tester2")

        prev = state.get("previous_sessions") or []
        assert isinstance(prev, list)
        assert len(prev) >= 1

    def test_state_contains_core_fields(self):
        agent = ParquetQueryAgent()
        state = agent.run_query(query="Top regions by sales", session_id="sess-xyz", user_id="tester3")

        # Core state fields exist
        assert "user_input" in state
        assert "session_id" in state
        assert "timestamp" in state
        assert "plan" in state or "plan_quality" in state
        assert "execution_result" in state or "execution_stats" in state
        assert "satisfaction" in state or "evaluator_notes" in state
        assert "final_output" in state


