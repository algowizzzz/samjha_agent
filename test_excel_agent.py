"""
Simple test script for Excel Agent.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from external.agent.excel_agent import handle_excel_query


def test_list_files():
    """Test listing Excel files"""
    print("=" * 60)
    print("Test 1: List Excel Files")
    print("=" * 60)
    
    result = handle_excel_query(
        user_query="List all Excel files",
        agent_data_folder=None,  # Will search all folders
        max_iterations=3
    )
    
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nObservations: {len(result['observations'])} actions executed")
    
    return result


def test_read_sheet(agent_data_folder: str = None):
    """Test reading an Excel sheet"""
    print("\n" + "=" * 60)
    print("Test 2: Read Excel Sheet")
    print("=" * 60)
    
    result = handle_excel_query(
        user_query="Show me the data in the Excel files",
        agent_data_folder=agent_data_folder,
        max_iterations=5
    )
    
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nAnswer:\n{result['answer']}")
    
    return result


if __name__ == "__main__":
    print("Excel Agent Test")
    print("=" * 60)
    
    # Test 1: List files
    try:
        result1 = test_list_files()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Read sheet (optional - specify folder if you have Excel files)
    # Uncomment and set folder if you have test data:
    # try:
    #     result2 = test_read_sheet(agent_data_folder="YourExcelFolder")
    # except Exception as e:
    #     print(f"Test 2 failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Tests completed!")

