#!/usr/bin/env python3
"""
Multi-turn follow-up test: 5 queries where Query 5 references Query 2.
Tests conversation_history (5 turns) vs prior_query_spec (latest only).
"""

from external.agent.parquet_agent import handle_query
from tools.tools_registry import ToolsRegistry
from pathlib import Path
import duckdb
import json

# Load tools
registry = ToolsRegistry()
tools_config_dir = Path("config/tools")
tool_names = [
    "list_dir", "inspect_table", "preview_rows", "search_glossary",
    "nl_to_sql_planner", "sql_plan_updater", "query_safety_validator",
    "execute_sql", "query_result_evaluator"
]
for tool_name in tool_names:
    tool_config = tools_config_dir / f"{tool_name}.json"
    if tool_config.exists():
        try:
            registry.load_tool_from_config(tool_config)
        except Exception as e:
            pass

print("=" * 80)
print("MULTI-TURN FOLLOW-UP TEST")
print("=" * 80)
print("Testing 5 queries where Query 5 references Query 2")
print("=" * 80)

# Query sequence with expected results
queries = [
    {
        "query": "Show me total revenue by region for orders in January 2024",
        "expected_rows": 4,
        "expected_values": ["East", "North", "South", "West"]
    },
    {
        "query": "What about breakdown by product category too?",
        "expected_rows": 7,
        "expected_values": ["Electronics", "Furniture"]  # Should have both categories
    },
    {
        "query": "Now filter to only Electronics category",
        "expected_rows": 4,
        "expected_values": ["Electronics"]  # Should only have Electronics
    },
    {
        "query": "Show me the top 3 products by sales quantity",
        "expected_rows": 3,
        "expected_values": ["Mouse", "Monitor", "Chair"]  # Top 3 by quantity
    },
    {
        "query": "What was the revenue for those categories we looked at earlier?",
        "expected_rows": 2,
        "expected_values": ["Electronics", "Furniture"]  # Should reference Query 2's categories
    }
]

# Track conversation history and prior states
conversation_history = []
prior_state = None
results = []

for i, query_info in enumerate(queries, 1):
    query = query_info["query"]
    expected_rows = query_info["expected_rows"]
    expected_values = query_info["expected_values"]
    
    print(f"\n{'='*80}")
    print(f"QUERY {i}: {query}")
    print("=" * 80)
    
    if i == 1:
        print("Expected: NEW_QUERY - revenue by region")
        print(f"Expected rows: {expected_rows} (regions: {', '.join(expected_values)})")
    elif i == 2:
        print("Expected: FOLLOW_UP - add category dimension")
        print(f"Expected rows: {expected_rows} (should have both categories: {', '.join(expected_values)})")
    elif i == 3:
        print("Expected: FOLLOW_UP - filter to Electronics")
        print(f"Expected rows: {expected_rows} (only {expected_values[0]} category)")
    elif i == 4:
        print("Expected: FOLLOW_UP - top 3 products by quantity")
        print(f"Expected rows: {expected_rows} (products: {', '.join(expected_values)})")
    elif i == 5:
        print("Expected: FOLLOW_UP - references Query 2's categories")
        print(f"Expected rows: {expected_rows} (categories: {', '.join(expected_values)})")
        print("  Key test: prior_query_spec is from Query 4, but should use")
        print("  conversation_history to understand 'those categories' = Query 2")
    
    print("-" * 80)
    
    # Execute query
    result = handle_query(
        user_query=query,
        conversation_history=conversation_history,
        prior_state=prior_state,
        tools_registry=registry,
        policy_limits={"max_attempts": 3, "max_rows": 1000, "timeout_seconds": 60, "allow_cross_join": False}
    )
    
    status = result.get('status')
    print(f"\nStatus: {status}")
    
    if status == 'SUCCESS':
        sql = result.get('final_sql', 'N/A')
        print(f"\n✓ SQL:\n{sql}")
        
        # Execute SQL to show results
        db_results = []
        conn = duckdb.connect(":memory:")
        try:
            sql_with_path = sql.replace('sample_sales_data', "read_csv_auto('mock_datawarehouse/ECommerce/sample_sales_data.csv')")
            db_results = conn.execute(sql_with_path).fetchall()
            
            print(f"\nResult ({len(db_results)} rows, expected {expected_rows}):")
            for row in db_results[:10]:  # Show first 10 rows
                print(f"  {row}")
            if len(db_results) > 10:
                print(f"  ... ({len(db_results) - 10} more rows)")
            
            # Verify row count
            if len(db_results) == expected_rows:
                print(f"  ✓ Row count matches expected ({expected_rows})")
            else:
                print(f"  ✗ Row count mismatch: got {len(db_results)}, expected {expected_rows}")
            
            # Verify expected values are present
            result_str = str(db_results).lower()
            all_found = True
            for val in expected_values:
                if val.lower() in result_str:
                    print(f"  ✓ Found expected value: {val}")
                else:
                    print(f"  ✗ Missing expected value: {val}")
                    all_found = False
                
        except Exception as e:
            print(f"Error executing SQL: {e}")
            db_results = []
        finally:
            conn.close()
        
        # Build conversation history entry
        conversation_history.append({
            "query": query,
            "sql": sql,
            "response": result.get('result_summary', ''),
            "status": "SUCCESS"
        })
        
        # Build prior_state for next query
        # Note: handle_query doesn't return query_spec directly, so we'll let
        # the Decider reconstruct it from conversation_history and prior_state
        # For follow-ups, the controller will pass the prior state which includes
        # the query_spec from the previous Decider output
        
        # For the first query, prior_state is None
        # For subsequent queries, we need to manually construct prior_state
        # based on what we know from the conversation
        
        # Since handle_query doesn't expose internal state, we'll use a simplified
        # prior_state that the Decider can work with
        if i == 1:
            # After Query 1, build initial prior_state
            prior_state = {
                "user_query": query,
                "conversation_history": conversation_history,
                "domain_md": "",
                "policy_limits": {"max_attempts": 3, "max_rows": 1000, "timeout_seconds": 60, "allow_cross_join": False},
                "query_spec": {
                    "business_question": query,
                    "dimensions": ["region"],
                    "metrics": [{"name": "revenue", "definition": "Sum of (quantity * price)"}],
                    "time": {"column": "order_date", "rule": "date_range", "start": "2024-01-01", "end": "2024-01-31"},
                    "start_table": {"name": "sample_sales_data", "path": "ECommerce/sample_sales_data.csv"}
                },
                "query_spec_status": {
                    "start_table_grain": {"status": "verified", "source": "tool_result", "blocks_execution": False},
                    "dimensions": {"status": "verified", "source": "user", "blocks_execution": False},
                    "metrics": {"status": "verified", "source": "user", "blocks_execution": False},
                    "time": {"status": "verified", "source": "user", "blocks_execution": False}
                },
                "last_executor_report": None,
                "attempt_count": 0
            }
        else:
            # For subsequent queries, update prior_state with latest query info
            # The Decider will merge this with conversation_history
            # We'll keep the prior_state structure but update user_query
            if prior_state:
                prior_state["user_query"] = query
                prior_state["conversation_history"] = conversation_history
        
        results.append({
            "query_num": i,
            "query": query,
            "status": status,
            "sql": sql,
            "result_count": len(db_results) if 'db_results' in locals() else 0,
            "expected_rows": expected_rows,
            "expected_values": expected_values
        })
        
        print(f"\n✓ Query {i} SUCCESS")
        print(f"  conversation_history length: {len(conversation_history)}")
        
    elif status == 'ASK_USER':
        print(f"\n? Asked User: {result.get('question', 'N/A')}")
        break
    else:
        print(f"\n✗ Query {i} FAILED")
        print(f"  Error Type: {result.get('error_type', 'N/A')}")
        print(f"  Reason: {result.get('reason', 'N/A')[:400]}")
        break

# Final summary
print(f"\n{'='*80}")
print("TEST SUMMARY")
print("=" * 80)
print(f"Total queries executed: {len(results)}")
print(f"Conversation history length: {len(conversation_history)}")

for r in results:
    print(f"\nQuery {r['query_num']}: {r['status']}")
    print(f"  Query: {r['query'][:60]}...")
    print(f"  Result rows: {r['result_count']} (expected: {r['expected_rows']})")
    if r['result_count'] == r['expected_rows']:
        print(f"  ✓ Row count correct")
    else:
        print(f"  ✗ Row count mismatch")

print(f"\n{'='*80}")
print("KEY VERIFICATION")
print("=" * 80)
if len(results) >= 5:
    print("✓ All 5 queries executed")
    print("\nQuery 5 Analysis:")
    q5_result = results[4]
    sql_lower = q5_result['sql'].lower()
    
    has_category = 'category' in sql_lower
    has_revenue = 'revenue' in sql_lower or 'sum(quantity * price)' in sql_lower
    has_january = '2024-01' in q5_result['sql']
    
    print(f"  Has 'category' dimension: {has_category} {'✓' if has_category else '✗'}")
    print(f"  Has 'revenue' metric: {has_revenue} {'✓' if has_revenue else '✗'}")
    print(f"  Has January 2024 filter: {has_january} {'✓' if has_january else '✗'}")
    
    if has_category and has_revenue and has_january:
        print("\n✓✓✓ Query 5 correctly referenced Query 2's category breakdown!")
    else:
        print("\n✗ Query 5 may not have correctly referenced Query 2")
else:
    print(f"✗ Only {len(results)} queries executed (expected 5)")

print("\n" + "=" * 80)

