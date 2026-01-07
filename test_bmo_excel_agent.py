"""
Test BMO Financials Excel Agent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from external.agent.bmo_excel_agent import handle_bmo_query


def test_bmo_agent():
    """Test BMO Excel agent with various queries"""
    
    print("=" * 80)
    print("BMO FINANCIALS EXCEL AGENT TEST")
    print("=" * 80)
    
    # Test 1: Understand the structure
    print("\n" + "-" * 80)
    print("TEST 1: Understand BMO Financials Structure")
    print("-" * 80)
    
    query1 = "What BMO Financials files do we have? Show me what they contain and their structure"
    
    print(f"\nQuery: {query1}\n")
    
    result1 = handle_bmo_query(
        user_query=query1,
        agent_data_folder="bmo_financials",
        max_iterations=10
    )
    
    print(f"Status: {result1['status']}")
    print(f"Iterations: {result1['iterations']}")
    print(f"\nAnswer:\n{result1['answer']}")
    
    # Test 2: Find specific metric (if user wants to explore)
    print("\n" + "=" * 80)
    print("TEST 2: Explore Index to Understand Available Data")
    print("=" * 80)
    
    query2 = "Read the Index sheet from one of the BMO files to see what financial information is available"
    
    print(f"\nQuery: {query2}\n")
    
    result2 = handle_bmo_query(
        user_query=query2,
        agent_data_folder="bmo_financials",
        max_iterations=10
    )
    
    print(f"Status: {result2['status']}")
    print(f"Iterations: {result2['iterations']}")
    print(f"\nAnswer:\n{result2['answer']}")
    
    print("\n" + "=" * 80)
    print("✅ Tests Complete")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_bmo_agent()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

