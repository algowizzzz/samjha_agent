"""
Test each agent node individually to verify config prompts are working
"""
import sys
import json
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from agent.config import QueryAgentConfig
from agent.state_manager import AgentStateManager
from agent.graph_nodes import (
    invoke_node,
    planner_node,
    clarify_node,
    replan_node,
    execute_node,
    evaluate_node,
    end_node
)
from agent.schemas import AgentState

# Initialize
cfg = QueryAgentConfig()
sm = AgentStateManager()

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def test_invoke_node():
    """Test invoke_node"""
    print("\n" + "="*80)
    print("TEST 1: INVOKE NODE")
    print("="*80)
    
    state: AgentState = {
        "user_input": "total sales",
        "user_id": "test_user",
        "session_id": "test-session-invoke",
        "timestamp": _now_iso(),
        "logs": []
    }
    
    try:
        result = invoke_node(state, cfg, sm)
        print("✅ invoke_node completed")
        print(f"   - Tables loaded: {len(result.get('table_schema', {}))}")
        print(f"   - Docs meta entries: {len(result.get('docs_meta', []))}")
        print(f"   - Control: {result.get('control')}")
        return result
    except Exception as e:
        print(f"❌ invoke_node failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_planner_node(state: AgentState):
    """Test planner_node"""
    print("\n" + "="*80)
    print("TEST 2: PLANNER NODE")
    print("="*80)
    
    try:
        result = planner_node(state, cfg)
        print("✅ planner_node completed")
        print(f"   - Plan quality: {result.get('plan_quality')}")
        print(f"   - Control: {result.get('control')}")
        plan = result.get('plan', {})
        if plan:
            print(f"   - SQL: {plan.get('sql', 'N/A')[:80]}...")
        return result
    except Exception as e:
        print(f"❌ planner_node failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_clarify_node(state: AgentState):
    """Test clarify_node"""
    print("\n" + "="*80)
    print("TEST 3: CLARIFY NODE")
    print("="*80)
    
    # Set up state for clarification
    state["plan_quality"] = "low"
    state["plan_explain"] = "Query is ambiguous"
    state["clarification_questions"] = ["Which table?", "What timeframe?"]
    state["clarify_reasoning"] = ["Need to know which dataset", "Time range is unclear"]
    
    try:
        result = clarify_node(state, cfg)
        print("✅ clarify_node completed")
        print(f"   - Control: {result.get('control')}")
        print(f"   - Clarify prompt length: {len(result.get('clarify_prompt', ''))}")
        print(f"   - Questions: {len(result.get('clarify_questions', []))}")
        return result
    except Exception as e:
        print(f"❌ clarify_node failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_execute_node(state: AgentState):
    """Test execute_node"""
    print("\n" + "="*80)
    print("TEST 4: EXECUTE NODE")
    print("="*80)
    
    # Ensure we have a plan
    if not state.get("plan"):
        print("   ⚠️  No plan in state, running planner first...")
        planner_result = planner_node(state, cfg)
        state.update(planner_result)
    
    try:
        result = execute_node(state, cfg)
        print("✅ execute_node completed")
        exec_result = result.get('execution_result', {})
        print(f"   - Rows returned: {exec_result.get('row_count', 0)}")
        print(f"   - Columns: {exec_result.get('columns', [])}")
        print(f"   - Control: {result.get('control')}")
        return result
    except Exception as e:
        print(f"❌ execute_node failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_evaluate_node(state: AgentState):
    """Test evaluate_node"""
    print("\n" + "="*80)
    print("TEST 5: EVALUATE NODE")
    print("="*80)
    
    # Ensure we have execution result
    if not state.get("execution_result"):
        print("   ⚠️  No execution result, running execute first...")
        if not state.get("plan"):
            planner_result = planner_node(state, cfg)
            state.update(planner_result)
        exec_result = execute_node(state, cfg)
        state.update(exec_result)
    
    try:
        result = evaluate_node(state, cfg)
        print("✅ evaluate_node completed")
        print(f"   - Satisfaction: {result.get('satisfaction')}")
        print(f"   - Control: {result.get('control')}")
        print(f"   - Evaluator notes: {result.get('evaluator_notes', 'N/A')[:100]}...")
        return result
    except Exception as e:
        print(f"❌ evaluate_node failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_end_node(state: AgentState):
    """Test end_node"""
    print("\n" + "="*80)
    print("TEST 6: END NODE")
    print("="*80)
    
    # Ensure we have all required state
    if not state.get("execution_result"):
        print("   ⚠️  Setting up required state...")
        if not state.get("plan"):
            planner_result = planner_node(state, cfg)
            state.update(planner_result)
        exec_result = execute_node(state, cfg)
        state.update(exec_result)
        eval_result = evaluate_node(state, cfg)
        state.update(eval_result)
    
    # Add metrics
    if not state.get("metrics"):
        state["metrics"] = {
            "start_time": _now_iso(),
            "clarify_turns": 0
        }
    
    try:
        result = end_node(state, cfg)
        print("✅ end_node completed")
        final_output = result.get('final_output', {})
        print(f"   - Response length: {len(final_output.get('response', ''))}")
        print(f"   - Raw table rows: {final_output.get('raw_table', {}).get('row_count', 0)}")
        prompt_monitor = final_output.get('prompt_monitor', {})
        if isinstance(prompt_monitor, dict) and 'procedural_reasoning' in prompt_monitor:
            print(f"   - Prompt monitor (LLM): {len(prompt_monitor['procedural_reasoning'])} chars")
        else:
            print(f"   - Prompt monitor (template): {type(prompt_monitor)}")
        return result
    except Exception as e:
        print(f"❌ end_node failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_config_prompts():
    """Test that all config prompts are accessible"""
    print("\n" + "="*80)
    print("TEST 0: CONFIG PROMPTS ACCESSIBILITY")
    print("="*80)
    
    prompts_to_check = [
        "planner_system",
        "planner_user_template",
        "clarify_template",
        "evaluator_system",
        "evaluator_user_template",
        "end_response_system",
        "end_response_user_template",
        "end_prompt_monitor_system",
        "end_prompt_monitor_user_template"
    ]
    
    all_ok = True
    for prompt_key in prompts_to_check:
        try:
            prompt = cfg.get_nested("prompts", prompt_key, default=None)
            if prompt:
                print(f"✅ {prompt_key}: {len(prompt)} chars")
            else:
                print(f"⚠️  {prompt_key}: Not found (will use default)")
        except Exception as e:
            print(f"❌ {prompt_key}: Error - {e}")
            all_ok = False
    
    return all_ok

def main():
    print("="*80)
    print("AGENT NODES INDIVIDUAL TESTING")
    print("="*80)
    
    # Test config first
    config_ok = test_config_prompts()
    if not config_ok:
        print("\n❌ Config test failed. Aborting.")
        return
    
    # Test nodes in sequence
    state = test_invoke_node()
    if not state:
        print("\n❌ Invoke node failed. Cannot continue.")
        return
    
    # Update state with invoke results
    test_state: AgentState = {
        "user_input": "total sales",
        "user_id": "test_user",
        "session_id": "test-session-full",
        "timestamp": _now_iso(),
        "logs": []
    }
    test_state.update(state)
    
    # Test planner
    planner_result = test_planner_node(test_state)
    if planner_result:
        test_state.update(planner_result)
    
    # Test clarify (with low quality plan)
    clarify_state = test_state.copy()
    clarify_state["plan_quality"] = "low"
    clarify_result = test_clarify_node(clarify_state)
    
    # Test execute (with good plan)
    exec_result = test_execute_node(test_state)
    if exec_result:
        test_state.update(exec_result)
    
    # Test evaluate
    eval_result = test_evaluate_node(test_state)
    if eval_result:
        test_state.update(eval_result)
    
    # Test end
    end_result = test_end_node(test_state)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✅ All nodes tested individually")
    print("\nTo test full flow, use: python test_agent_interactive.py")

if __name__ == "__main__":
    main()

