"""
Test Excel Agent with insights and analysis queries
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from external.agent.excel_agent import handle_excel_query
import json


def print_query_result(query_num, query, result):
    """Print formatted query and result"""
    print("\n" + "=" * 80)
    print(f"QUERY {query_num}")
    print("=" * 80)
    print(f"\nQuery: {query}\n")
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nAnswer:\n{result['answer']}\n")
    print("-" * 80)
    print(f"Actions executed: {len(result['observations'])}")
    for i, obs in enumerate(result['observations'][:3], 1):  # Show first 3 actions
        action = obs['action']
        result_data = obs['result']
        print(f"\n{i}. {action['tool']}")
        if isinstance(result_data, dict) and 'data' in result_data:
            print(f"   Read {len(result_data.get('data', []))} rows from {result_data.get('file_path', 'file')}")


def test_insights():
    """Test extracting insights from data"""
    print("\n" + "🔍 " * 20)
    print("TEST: Extract Insights from Data")
    print("🔍 " * 20)
    
    query1 = "Extract insights from the data. What patterns, trends, or interesting findings can you identify?"
    
    result1 = handle_excel_query(
        user_query=query1,
        agent_data_folder="ECommerce",
        max_iterations=15
    )
    
    print_query_result(1, query1, result1)
    return result1


def test_highest_purchase_value():
    """Test finding customer with highest purchases by dollar value"""
    print("\n" + "🔍 " * 20)
    print("TEST: Customer with Highest Purchases by Dollar Value")
    print("🔍 " * 20)
    
    query2 = "Which customer has the highest total purchases by dollar value? Show me the customer name, their tier, and total spending amount"
    
    result2 = handle_excel_query(
        user_query=query2,
        agent_data_folder="ECommerce",
        max_iterations=15
    )
    
    print_query_result(2, query2, result2)
    return result2


def test_highest_purchase_count_by_month():
    """Test finding customer with highest purchases by count over each month"""
    print("\n" + "🔍 " * 20)
    print("TEST: Customer with Highest Purchases by Count Over Each Month")
    print("🔍 " * 20)
    
    query3 = "For each month, which customer has the highest number of purchases (by count)? Show me the month, customer name, and number of orders"
    
    result3 = handle_excel_query(
        user_query=query3,
        agent_data_folder="ECommerce",
        max_iterations=15
    )
    
    print_query_result(3, query3, result3)
    return result3


if __name__ == "__main__":
    try:
        # Test 1: Extract insights
        result1 = test_insights()
        
        # Test 2: Highest purchase value
        result2 = test_highest_purchase_value()
        
        # Test 3: Highest purchase count by month
        result3 = test_highest_purchase_count_by_month()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

