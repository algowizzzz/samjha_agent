"""
Analyze BMO Financials folder using Excel Agent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from external.agent.excel_agent import handle_excel_query


def analyze_bmo_financials():
    """Analyze the BMO financials folder"""
    print("\n" + "=" * 80)
    print("BMO FINANCIALS ANALYSIS")
    print("=" * 80)
    
    # Query 1: Understand what files we have
    query1 = "List all Excel files and show me what data is available in the bmo_financials folder. What do these files contain and how are they structured?"
    
    print("\n" + "-" * 80)
    print("QUERY 1: Understand the files")
    print("-" * 80)
    print(f"Query: {query1}\n")
    
    result1 = handle_excel_query(
        user_query=query1,
        agent_data_folder="bmo_financials",
        max_iterations=15
    )
    
    print(f"Status: {result1['status']}")
    print(f"Iterations: {result1['iterations']}")
    print(f"\nAnswer:\n{result1['answer']}")
    
    return result1


if __name__ == "__main__":
    try:
        result = analyze_bmo_financials()
        print("\n" + "=" * 80)
        print("Analysis Complete")
        print("=" * 80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

