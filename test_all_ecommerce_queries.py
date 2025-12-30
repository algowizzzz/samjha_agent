#!/usr/bin/env python3
"""
Test all 10 ecommerce_advanced queries with detailed logging.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from tools.tools_registry import ToolsRegistry
from external.agent.parquet_agent import handle_query

# Setup logging
log_file = Path("test_ecommerce_queries_log.json")
results = {
    "test_run_time": datetime.now().isoformat(),
    "queries": []
}

# Test queries
queries = [
    "top products by revenue",
    "revenue by region and category for January",
    "inventory stock value month over month",
    "top 5 customers by total purchases",
    "sales revenue by category for Electronics in North region",
    "average order value by customer tier",
    "products with low stock (below reorder level) in March",
    "revenue by region month over month",
    "sales revenue and order count by category",
    "total stock value by supplier",
]

# Initialize tools registry
print("=" * 80)
print("INITIALIZING TEST SUITE")
print("=" * 80)
reg = ToolsRegistry()
for cfg in Path('external/config/tools').glob('*.json'):
    if cfg.stem in ['catalog_data','list_dir','inspect_table','preview_rows','search_glossary',
                    'nl_to_sql_planner','sql_plan_updater','query_safety_validator','execute_sql','query_result_evaluator']:
        try:
            reg.load_tool_from_config(cfg)
        except Exception:
            pass

print(f"Loaded tools registry. Testing {len(queries)} queries.\n")

# Run each query
for idx, query in enumerate(queries, 1):
    print("=" * 80)
    print(f"TEST {idx}/{len(queries)}: {query}")
    print("=" * 80)
    
    query_result = {
        "query_id": idx,
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "status": None,
        "error": None,
        "sql": None,
        "aggregation_plan": None,
        "response_preview": None,
        "query_spec": None,
        "issues": []
    }
    
    try:
        # Run query
        result = handle_query(
            query, 
            [], 
            agent_id='ecommerce_advanced', 
            tools_registry=reg, 
            show_thinking=False
        )
        
        # Extract results
        query_result["status"] = result.get("status")
        query_result["sql"] = result.get("final_sql", "")
        query_result["response_preview"] = result.get("finished_output", "")[:300] if result.get("finished_output") else None
        
        # Check query_spec
        query_spec = result.get("query_spec", {})
        query_result["query_spec"] = {
            "business_question": query_spec.get("business_question"),
            "start_table": query_spec.get("start_table", {}),
            "dimensions": query_spec.get("dimensions", []),
            "metrics": query_spec.get("metrics", []),
            "aggregation_plan": query_spec.get("aggregation_plan"),
            "time": query_spec.get("time", {}),
        }
        
        # Check aggregation_plan
        agg_plan = query_spec.get("aggregation_plan")
        query_result["aggregation_plan"] = agg_plan
        
        # Validate results
        if query_result["status"] == "SUCCESS":
            # Check if SQL uses metric.definition correctly
            sql_lower = (query_result["sql"] or "").lower()
            if "sum(revenue)" in sql_lower and "sum(quantity * price)" not in sql_lower:
                query_result["issues"].append("SQL uses metric.name (revenue) instead of metric.definition (quantity * price)")
            if "sum(stock_value)" in sql_lower and "sum(stock_quantity * unit_cost)" not in sql_lower:
                query_result["issues"].append("SQL uses metric.name (stock_value) instead of metric.definition (stock_quantity * unit_cost)")
            
            # Check if aggregation_plan is present for multi-month queries
            if "month over month" in query.lower() or "month" in query.lower():
                if not agg_plan or (isinstance(agg_plan, str) and not agg_plan):
                    query_result["issues"].append("Missing aggregation_plan for multi-month query")
                elif isinstance(agg_plan, dict) and agg_plan.get("aggregation_type") != "union_all_then_group":
                    query_result["issues"].append(f"Aggregation plan type is {agg_plan.get('aggregation_type')}, expected union_all_then_group")
            
            # Check if SQL uses UNION ALL for multi-month
            if "month over month" in query.lower() and "union all" not in sql_lower:
                query_result["issues"].append("Multi-month query missing UNION ALL in SQL")
        
        # Print summary
        print(f"\nStatus: {query_result['status']}")
        if query_result["sql"]:
            print(f"SQL (first 400 chars):\n{query_result['sql'][:400]}...")
        if agg_plan:
            print(f"Aggregation Plan: {json.dumps(agg_plan, default=str)[:200]}...")
        if query_result["issues"]:
            print(f"Issues: {query_result['issues']}")
        
    except Exception as e:
        query_result["status"] = "ERROR"
        query_result["error"] = str(e)
        import traceback
        query_result["traceback"] = traceback.format_exc()
        print(f"ERROR: {e}")
    
    results["queries"].append(query_result)
    print()  # Blank line between tests

# Save results
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)

success_count = sum(1 for q in results["queries"] if q["status"] == "SUCCESS")
error_count = sum(1 for q in results["queries"] if q["status"] == "ERROR")
ask_user_count = sum(1 for q in results["queries"] if q["status"] == "ASK_USER")

print(f"\nTotal queries: {len(queries)}")
print(f"  SUCCESS: {success_count}")
print(f"  ERROR: {error_count}")
print(f"  ASK_USER: {ask_user_count}")

# Check aggregation_plan presence
agg_plan_count = sum(1 for q in results["queries"] if q.get("aggregation_plan"))
print(f"\nQueries with aggregation_plan: {agg_plan_count}/{len(queries)}")

# Check for issues
total_issues = sum(len(q.get("issues", [])) for q in results["queries"])
print(f"Total issues found: {total_issues}")

if total_issues > 0:
    print("\nIssues by query:")
    for q in results["queries"]:
        if q.get("issues"):
            print(f"  {q['query_id']}. {q['query']}: {q['issues']}")

# Save to file
with open(log_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nDetailed results saved to: {log_file}")

# Print quick summary table
print("\n" + "=" * 80)
print("QUICK SUMMARY TABLE")
print("=" * 80)
print(f"{'ID':<4} {'Status':<12} {'Has SQL':<10} {'Has AggPlan':<15} {'Issues':<10}")
print("-" * 80)
for q in results["queries"]:
    has_sql = "✓" if q.get("sql") else "✗"
    has_agg = "✓" if q.get("aggregation_plan") else "✗"
    issues = len(q.get("issues", []))
    status = q.get("status", "UNKNOWN")
    print(f"{q['query_id']:<4} {status:<12} {has_sql:<10} {has_agg:<15} {issues:<10}")

print("\n" + "=" * 80)

