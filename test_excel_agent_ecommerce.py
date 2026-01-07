"""
Test Excel Agent with ECommerce folder
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from external.agent.excel_agent import handle_excel_query


def test_ecommerce_query():
    """Test Excel agent on ECommerce folder"""
    print("=" * 70)
    print("Excel Agent Test - ECommerce Folder")
    print("=" * 70)
    
    query = "List all Excel files in the ECommerce folder and show me what data is available"
    
    print(f"\nQuery: {query}\n")
    print("-" * 70)
    
    result = handle_excel_query(
        user_query=query,
        agent_data_folder="ECommerce",
        max_iterations=5
    )
    
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("\n" + "-" * 70)
    print("Observation Details:")
    print("-" * 70)
    for i, obs in enumerate(result['observations'], 1):
        print(f"\nObservation {i}:")
        print(f"  Action: {obs['action']['tool']}")
        print(f"  Result keys: {list(obs['result'].keys()) if isinstance(obs['result'], dict) else type(obs['result'])}")
        if isinstance(obs['result'], dict) and 'files' in obs['result']:
            files = obs['result']['files']
            print(f"  Files found: {len(files)}")
            for f in files[:3]:
                print(f"    - {f['name']}")
        elif isinstance(obs['result'], dict) and 'sheets' in obs['result']:
            print(f"  Sheets: {obs['result']['sheets']}")
        elif isinstance(obs['result'], dict) and 'data' in obs['result']:
            data_count = len(obs['result'].get('data', []))
            print(f"  Data rows: {data_count}")
            print(f"  Columns: {', '.join(obs['result'].get('columns', [])[:5])}")
    
    print("\n" + "=" * 70)
    
    return result


if __name__ == "__main__":
    try:
        result = test_ecommerce_query()
        print("✓ Test completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

