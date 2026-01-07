"""
Test Excel Agent with multi-file queries and relationships
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from external.agent.excel_agent import handle_excel_query


def test_all_files_and_relationships():
    """Test reading all files and understanding relationships"""
    print("=" * 80)
    print("Test 1: Read ALL files and show relationships")
    print("=" * 80)
    
    query = "4 Excel files exist with names. Read them all and provide basic details about what each contains and how they are connected"
    
    print(f"\nQuery: {query}\n")
    print("-" * 80)
    
    result = handle_excel_query(
        user_query=query,
        agent_data_folder="ECommerce",
        max_iterations=15
    )
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\n" + "=" * 80)
    
    return result


def test_complex_query():
    """Test a complex query involving multiple files"""
    print("\n" + "=" * 80)
    print("Test 2: Complex Multi-File Query")
    print("=" * 80)
    
    query = "Which customers have purchased Electronics products, and what is their total spending? Show me customer names, their tier, and total amount spent"
    
    print(f"\nQuery: {query}\n")
    print("-" * 80)
    
    result = handle_excel_query(
        user_query=query,
        agent_data_folder="ECommerce",
        max_iterations=15
    )
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\n" + "=" * 80)
    
    return result


if __name__ == "__main__":
    try:
        # Test 1: Read all files and relationships
        result1 = test_all_files_and_relationships()
        
        # Test 2: Complex query
        result2 = test_complex_query()
        
        print("\n✓ All tests completed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

