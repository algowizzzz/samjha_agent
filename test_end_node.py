"""
Test script for end_node - independent testing
Tests the end_node function with sample state to verify LLM integration
"""
import sys
import json
from datetime import datetime
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from agent.graph_nodes import end_node
from agent.config import QueryAgentConfig
from agent.schemas import AgentState

# ============================================================================
# SAMPLE STATE - Modify as needed for testing
# ============================================================================
def create_sample_state():
    """Create a sample state object for testing"""
    return {
        "user_input": "total sales by region",
        "user_id": "admin",
        "session_id": "test-session-001",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        
        # Enrichment
        "docs_meta": [
            {
                "table": "sample_sales_data",
                "description": "Sales transactions data containing order information, products sold, pricing, and regional data",
                "business_context": "This table tracks all sales orders. Use 'price' column for sales amount/revenue. Use 'quantity' for units sold.",
                "key_columns": {
                    "order_id": "Unique identifier for each order",
                    "customer_id": "Reference to customer who placed the order",
                    "product": "Name of the product sold",
                    "category": "Product category (e.g., Electronics, Furniture)",
                    "quantity": "Number of units sold",
                    "price": "Price per unit (use this for sales amount/revenue calculations)",
                    "order_date": "Date when the order was placed",
                    "region": "Geographic region where the sale occurred (North, South, East, West)"
                }
            },
            {
                "type": "business_glossary",
                "glossary": {
                    "sales_amount": "Use 'price' column from sample_sales_data",
                    "revenue": "Use 'price' column from sample_sales_data",
                    "total_sales": "SUM(price) from sample_sales_data",
                    "units_sold": "Use 'quantity' column from sample_sales_data"
                }
            }
        ],
        "table_schema": {
            "sample_sales_data": {
                "columns": [
                    {"name": "order_id", "type": "VARCHAR"},
                    {"name": "customer_id", "type": "VARCHAR"},
                    {"name": "product", "type": "VARCHAR"},
                    {"name": "category", "type": "VARCHAR"},
                    {"name": "quantity", "type": "VARCHAR"},
                    {"name": "price", "type": "VARCHAR"},
                    {"name": "order_date", "type": "VARCHAR"},
                    {"name": "region", "type": "VARCHAR"}
                ]
            }
        },
        "parquet_location": "data/duckdb",
        
        # Planning
        "plan": {
            "sql": "SELECT region, SUM(CAST(price AS FLOAT)) as total_sales FROM sample_sales_data GROUP BY region ORDER BY total_sales DESC LIMIT 10",
            "type": "sql_plan",
            "target_table": "sample_sales_data"
        },
        "plan_quality": "high",
        "plan_explain": "Query groups sales by region and sums the price column to calculate total sales per region",
        "clarification_questions": [],
        
        # Execution
        "execution_result": {
            "columns": ["region", "total_sales"],
            "rows": [
                {"region": "North", "total_sales": 4525.0},
                {"region": "South", "total_sales": 3200.0},
                {"region": "East", "total_sales": 2800.0},
                {"region": "West", "total_sales": 2100.0}
            ],
            "row_count": 4,
            "query": "SELECT region, SUM(CAST(price AS FLOAT)) as total_sales FROM sample_sales_data GROUP BY region ORDER BY total_sales DESC LIMIT 10",
            "execution_time_ms": 15.5
        },
        "execution_stats": {
            "error": None,
            "execution_time_ms": 15.5,
            "limited": False
        },
        
        # Control
        "control": "end",
        "last_node": "evaluate",
        
        # Evaluation
        "satisfaction": "satisfied",
        "evaluator_notes": "Query executed successfully and returned results that match the user's request for total sales by region",
        
        # Memory
        "conversation_history": [],
        
        # Telemetry
        "logs": [
            {"node": "invoke", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": "invoke started"},
            {"node": "planner", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": "plan generated"},
            {"node": "execute", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": "query executed successfully"},
            {"node": "evaluate", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": "evaluation complete"}
        ],
        
        # Metrics
        "metrics": {
            "start_time": datetime.utcnow().isoformat() + "Z",
            "plan_id": "test-plan-001",
            "clarify_turns": 0,
            "node_timings_ms": {
                "planner": 2000.0,
                "execute": 15.5,
                "evaluate": 1500.0
            }
        }
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*80)
    print("TESTING END_NODE")
    print("="*80)
    
    # Create sample state
    state = create_sample_state()
    print(f"\n📊 Sample State Created:")
    print(f"  - User Query: {state['user_input']}")
    print(f"  - Plan Quality: {state['plan_quality']}")
    print(f"  - Execution Rows: {state['execution_result']['row_count']}")
    print(f"  - Satisfaction: {state['satisfaction']}")
    
    # Load config
    cfg = QueryAgentConfig()
    
    # Call end_node
    print("\n🔄 Calling end_node...")
    try:
        result = end_node(state, cfg)
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        # Display raw table
        if result.get("raw_table"):
            raw_table = result["raw_table"]
            print(f"\n📋 RAW TABLE:")
            print(f"  Columns: {raw_table.get('columns', [])}")
            print(f"  Row Count: {raw_table.get('row_count', 0)}")
            print(f"  First 3 rows:")
            for i, row in enumerate(raw_table.get("rows", [])[:3], 1):
                print(f"    {i}. {row}")
        
        # Display response
        final_output = result.get("final_output", {})
        if final_output.get("response"):
            print(f"\n💬 RESPONSE (LLM-generated):")
            print("-" * 80)
            print(final_output["response"])
            print("-" * 80)
        
        # Display prompt monitor
        if final_output.get("prompt_monitor"):
            prompt_monitor = final_output["prompt_monitor"]
            print(f"\n🔍 PROMPT MONITOR (LLM-generated):")
            print("-" * 80)
            if isinstance(prompt_monitor, dict) and "procedural_reasoning" in prompt_monitor:
                print(prompt_monitor["procedural_reasoning"])
            else:
                print(json.dumps(prompt_monitor, indent=2))
            print("-" * 80)
        
        # Display full result as JSON
        print(f"\n📄 FULL RESULT (JSON):")
        print("="*80)
        print(json.dumps(result, indent=2, default=str))
        
        # Check if LLM was used
        logs = result.get("logs", [])
        llm_logs = [log for log in logs if "LLM" in log.get("msg", "")]
        if llm_logs:
            print(f"\n✅ LLM Usage Detected:")
            for log in llm_logs:
                print(f"  - {log.get('msg')}")
        else:
            print(f"\n⚠️  No LLM usage detected (may have fallen back to templates)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

