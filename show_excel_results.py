"""
Show detailed Excel Agent query and results
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from external.agent.excel_agent import handle_excel_query
import json


def print_section(title, content):
    """Print a formatted section"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(content)


def show_query_and_results():
    """Display queries and detailed results"""
    
    # Query 1: Read all files and relationships
    print("\n" + "🔍 " * 20)
    print("QUERY 1: Read All Files and Relationships")
    print("🔍 " * 20)
    
    query1 = "4 Excel files exist with names. Read them all and provide basic details about what each contains and how they are connected"
    
    print_section("QUERY", query1)
    
    result1 = handle_excel_query(
        user_query=query1,
        agent_data_folder="ECommerce",
        max_iterations=15
    )
    
    print_section("RESULT", f"""
Status: {result1['status']}
Iterations: {result1['iterations']}

Answer:
{result1['answer']}
""")
    
    print_section("OBSERVATIONS", f"Total actions executed: {len(result1['observations'])}")
    for i, obs in enumerate(result1['observations'], 1):
        action = obs['action']
        result = obs['result']
        print(f"\n{i}. Action: {action['tool']}")
        print(f"   Arguments: {json.dumps(action.get('arguments', {}), indent=2)}")
        if isinstance(result, dict):
            if 'files' in result:
                print(f"   Found {len(result['files'])} file(s)")
            if 'sheets' in result:
                print(f"   Sheets: {result['sheets']}")
            if 'data' in result:
                print(f"   Data rows: {len(result.get('data', []))}")
                print(f"   Columns: {', '.join(result.get('columns', [])[:5])}...")
            if 'error' in result:
                print(f"   Error: {result['error']}")
    
    # Query 2: Complex multi-file query
    print("\n\n" + "🔍 " * 20)
    print("QUERY 2: Complex Multi-File Analysis")
    print("🔍 " * 20)
    
    query2 = "Which customers have purchased Electronics products, and what is their total spending? Show me customer names, their tier, and total amount spent"
    
    print_section("QUERY", query2)
    
    result2 = handle_excel_query(
        user_query=query2,
        agent_data_folder="ECommerce",
        max_iterations=15
    )
    
    print_section("RESULT", f"""
Status: {result2['status']}
Iterations: {result2['iterations']}

Answer:
{result2['answer']}
""")
    
    print_section("OBSERVATIONS", f"Total actions executed: {len(result2['observations'])}")
    for i, obs in enumerate(result2['observations'], 1):
        action = obs['action']
        result = obs['result']
        print(f"\n{i}. Action: {action['tool']}")
        print(f"   Arguments: {json.dumps(action.get('arguments', {}), indent=2)}")
        if isinstance(result, dict):
            if 'files' in result:
                files = result['files']
                print(f"   Found {len(files)} file(s):")
                for f in files[:3]:
                    print(f"     - {f['name']} ({f['size_bytes']} bytes)")
            if 'sheets' in result:
                print(f"   Sheets: {result['sheets']}")
            if 'data' in result:
                data = result.get('data', [])
                cols = result.get('columns', [])
                rows = result.get('row_count', 0)
                print(f"   Total rows: {rows}, Preview rows: {len(data)}")
                print(f"   Columns ({len(cols)}): {', '.join(cols)}")
                if data:
                    print(f"   Sample row: {dict(list(data[0].items())[:3])}")
            if 'error' in result:
                print(f"   Error: {result['error']}")


if __name__ == "__main__":
    show_query_and_results()

