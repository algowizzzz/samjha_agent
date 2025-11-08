"""
Interactive agent test script - change the query and run
Output as JSON for easy inspection
"""
import sys
import json
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from agent.parquet_agent import ParquetQueryAgent

# ============================================================================
# CHANGE YOUR QUERY HERE
# ============================================================================
QUERY = "regional revenue"
USER_ID = "admin"

# ============================================================================
# RESUME CLARIFICATION (optional)
# Set RESUME_SESSION_ID to the session_id from a previous run that requested clarification
# Set USER_CLARIFICATION to the user's response
# ============================================================================
RESUME_SESSION_ID = None # e.g., '74081b75-8446-46dc-b93d-e58a2371e740'
USER_CLARIFICATION = None # e.g., 'I want the top 5 sales ordered by price column'

# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("\n" + "="*80)
print(f"QUERY: {QUERY}")
print("="*80)

agent = ParquetQueryAgent()
result = {}

if RESUME_SESSION_ID and USER_CLARIFICATION:
    print("\n🔄 Resuming query with clarification")
    result = agent.resume_with_clarification(
        session_id=RESUME_SESSION_ID,
        user_clarification=USER_CLARIFICATION,
        user_id=USER_ID
    )
else:
    print("\n🆕 Starting new query")
    result = agent.run_query(QUERY, user_id=USER_ID)

print("\n" + "="*80)
print("RESULT (JSON)")
print("="*80)

# Prepare a simplified result for JSON output
output = {
    "session_id": result.get("session_id"),
    "control": result.get("control"),
    "plan_quality": result.get("plan_quality"),
    "plan_sql": result.get("plan", {}).get("sql", "N/A"),
    "plan_explain": result.get("plan_explain", "N/A"),
}

# Clarification info
if result.get('clarify_prompt'):
    output["clarification"] = {
        "prompt": result.get("clarify_prompt"),
        "questions": result.get("clarify_questions"),
        "reasoning": result.get("clarify_reasoning")
    }

# Execution info
exec_result = result.get("execution_result")
if exec_result:
    output["execution"] = {
        "row_count": exec_result.get("row_count", 0),
        "columns": exec_result.get("columns", []),
        "sample_rows": exec_result.get("rows", [])[:3],  # First 3 rows
        "execution_time_ms": result.get("execution_stats", {}).get("execution_time_ms", 0),
        "error": result.get("execution_stats", {}).get("error")
    }

# Final output
final_output = result.get('final_output', {})
if final_output:
    output["final_output"] = {
        "response": final_output.get("response", "N/A"),
        "raw_table": final_output.get("raw_table", {}),
        "prompt_monitor": final_output.get("prompt_monitor", {}),
        "satisfaction": result.get("satisfaction")
    }

# Logs
output["logs"] = result.get('logs', [])[-10:]  # Last 10 logs

# Metrics
output["metrics"] = result.get('metrics', {})

print(json.dumps(output, indent=2, ensure_ascii=False))

print("\n" + "="*80)
print("QUICK SUMMARY")
print("="*80)
print(f"Session ID: {output['session_id']}")
print(f"Control: {output['control']}")
print(f"Plan Quality: {output['plan_quality']}")

if output.get('clarification'):
    print(f"\n✋ CLARIFICATION NEEDED:")
    print(f"   Questions: {len(output['clarification']['questions'])}")
    print(f"   Reasoning: {len(output['clarification']['reasoning'])}")
    print(f"\n   To resume, set:")
    print(f"   RESUME_SESSION_ID = '{output['session_id']}'")
    print(f"   USER_CLARIFICATION = 'your answer here'")

if output.get('execution'):
    print(f"\n✅ EXECUTION:")
    print(f"   Rows: {output['execution']['row_count']}")
    print(f"   Time: {output['execution']['execution_time_ms']}ms")

# Show final output
if final_output:
    print(f"\n📊 FINAL RESPONSE (LLM-generated):")
    print("="*80)
    response = final_output.get('response', 'N/A')
    print(response)
    print("\n" + "="*80)
    
    # Show raw table if available
    raw_table = final_output.get('raw_table')
    if raw_table:
        print(f"\n📋 RAW TABLE:")
        print("="*80)
        print(f"Columns: {raw_table.get('columns', [])}")
        print(f"Row Count: {raw_table.get('row_count', 0)}")
        rows = raw_table.get('rows', [])[:5]
        if rows:
            for i, row in enumerate(rows, 1):
                print(f"{i}. {row}")
    
    # Show prompt monitor if available
    prompt_monitor = final_output.get('prompt_monitor')
    if prompt_monitor:
        print(f"\n🔍 PROMPT MONITOR (LLM-generated):")
        print("="*80)
        if isinstance(prompt_monitor, dict) and 'procedural_reasoning' in prompt_monitor:
            print(prompt_monitor['procedural_reasoning'][:500] + "..." if len(prompt_monitor['procedural_reasoning']) > 500 else prompt_monitor['procedural_reasoning'])
        else:
            print("(Structured prompt monitor available)")
    
    # Show first few rows from execution if raw_table not shown
    if not raw_table and exec_result:
        rows = exec_result.get('rows', [])
        if rows:
            print(f"\n📋 DATA PREVIEW (first 5 rows):")
            print("="*80)
            for i, row in enumerate(rows[:5], 1):
                print(f"{i}. {row}")

print("\n" + "="*80)
