#!/usr/bin/env python3
"""
Test agent queries with proper environment setup
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), '.env'))

print("=" * 70)
print("TESTING AGENTS AFTER DOMAIN_MD PATH EXTRACTION CHANGES")
print("=" * 70)

from external.agent.parquet_agent import handle_query
from tools.tools_registry import ToolsRegistry

# Initialize tools registry
registry = ToolsRegistry()
tools_config_dir = Path("external/config/tools")
tool_names = [
    "catalog_data",
    "list_dir", "inspect_table", "preview_rows", "search_glossary",
    "nl_to_sql_planner", "sql_plan_updater", "query_safety_validator",
    "execute_sql", "query_result_evaluator"
]

print("\nLoading tools...")
for tool_name in tool_names:
    tool_config = tools_config_dir / f"{tool_name}.json"
    if tool_config.exists():
        try:
            registry.load_tool_from_config(tool_config)
        except Exception as e:
            print(f"  Warning: {tool_name}: {e}")

print("✅ Tools loaded\n")

# Test 1: ecommerce agent (flat structure)
print("=" * 70)
print("TEST 1: ecommerce agent (flat structure)")
print("Query: 'top products by sales'")
print("=" * 70)

try:
    result1 = handle_query(
        user_query="top products by sales",
        conversation_history=[],
        agent_id="ecommerce",
        tools_registry=registry,
        show_thinking=False
    )
    
    print(f"\nStatus: {result1.get('status')}")
    if result1.get('status') == 'SUCCESS':
        results = result1.get('results', {})
        rows = results.get('rows', [])
        schema = results.get('schema', {})
        print(f"✅ TEST 1 SUCCESS")
        print(f"   Rows: {len(rows)}")
        print(f"   Columns: {list(schema.keys())[:5]}")
        if rows:
            print(f"   Sample row: {rows[0]}")
    elif result1.get('status') == 'ERROR':
        print(f"❌ TEST 1 ERROR")
        print(f"   Reason: {result1.get('reason', 'Unknown')[:300]}")
        print(f"   Type: {result1.get('error_type', 'Unknown')}")
    elif result1.get('status') == 'ASK_USER':
        print(f"⚠️  TEST 1 ASK_USER")
        print(f"   Question: {result1.get('question', 'N/A')[:200]}")
        print(f"   Why: {result1.get('why_non_defaultable', 'N/A')[:200]}")
    else:
        print(f"   Status: {result1.get('status')}")
        print(f"   Result keys: {list(result1.keys())}")
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()

# Test 2: ecommerce_advanced agent (nested structure)
print("\n" + "=" * 70)
print("TEST 2: ecommerce_advanced agent (nested structure)")
print("Query: 'top products by sales for january'")
print("=" * 70)

try:
    result2 = handle_query(
        user_query="top products by sales for january",
        conversation_history=[],
        agent_id="ecommerce_advanced",
        tools_registry=registry,
        show_thinking=False
    )
    
    print(f"\nStatus: {result2.get('status')}")
    if result2.get('status') == 'SUCCESS':
        results = result2.get('results', {})
        rows = results.get('rows', [])
        schema = results.get('schema', {})
        print(f"✅ TEST 2 SUCCESS")
        print(f"   Rows: {len(rows)}")
        print(f"   Columns: {list(schema.keys())[:5]}")
        if rows:
            print(f"   Sample row: {rows[0]}")
    elif result2.get('status') == 'ERROR':
        print(f"❌ TEST 2 ERROR")
        print(f"   Reason: {result2.get('reason', 'Unknown')[:300]}")
        print(f"   Type: {result2.get('error_type', 'Unknown')}")
    elif result2.get('status') == 'ASK_USER':
        print(f"⚠️  TEST 2 ASK_USER")
        print(f"   Question: {result2.get('question', 'N/A')[:200]}")
        print(f"   Why: {result2.get('why_non_defaultable', 'N/A')[:200]}")
    else:
        print(f"   Status: {result2.get('status')}")
        print(f"   Result keys: {list(result2.keys())}")
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

