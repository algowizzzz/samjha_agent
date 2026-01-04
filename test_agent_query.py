#!/usr/bin/env python3
"""
Test agent query via backend API
Shows how to make a query and display results
"""

import requests
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:5000"
AGENT_ID = "financial_news_research_agent_for_bankers"

def test_query():
    """Test a query against the financial news agent"""
    
    # Test query
    test_query = "Find recent news and regulatory information about Apple Inc. for a credit risk assessment"
    
    print("=" * 60)
    print("Testing Financial News Agent Query")
    print("=" * 60)
    print(f"Agent ID: {AGENT_ID}")
    print(f"Query: {test_query}")
    print()
    
    # Check if Flask is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print("✅ Flask app is running")
    except requests.exceptions.ConnectionError:
        print("❌ Flask app is not running")
        print("   Please start it with: python app.py")
        return
    except Exception as e:
        print(f"⚠️  Error checking Flask: {e}")
    
    print()
    print("Making API request...")
    print(f"POST {BASE_URL}/api/agents/{AGENT_ID}/runs")
    print()
    
    # Prepare request
    payload = {
        "query": test_query,
        "conversation_id": None  # New conversation
    }
    
    try:
        # Note: This will fail without authentication, but we can show the structure
        response = requests.post(
            f"{BASE_URL}/api/agents/{AGENT_ID}/runs",
            json=payload,
            timeout=5,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            run_id = data.get("run_id")
            print(f"✅ Run started!")
            print(f"   Run ID: {run_id}")
            print()
            print("To get events, use SSE endpoint:")
            print(f"   GET {BASE_URL}/api/runs/{run_id}/events")
            return run_id
        elif response.status_code in [401, 403]:
            print("⚠️  Authentication required (expected)")
            print()
            print("To test with authentication:")
            print("1. Start Flask: python app.py")
            print("2. Log in via browser: http://localhost:5000")
            print("3. Open browser dev tools → Network tab")
            print("4. Make a query in the UI")
            print("5. Copy the session cookie")
            print("6. Use it in API requests")
            print()
            print("Or test directly in the UI:")
            print(f"   http://localhost:5000/agent/chat/{AGENT_ID}")
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   {json.dumps(error_data, indent=2)}")
            except:
                print(f"   {response.text[:200]}")
                
    except requests.exceptions.Timeout:
        print("⏱️  Request timed out (query may be processing)")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def show_query_structure():
    """Show the query structure and expected flow"""
    print()
    print("=" * 60)
    print("Query Execution Flow")
    print("=" * 60)
    print()
    print("1. POST /api/agents/{agent_id}/runs")
    print("   Body: { 'query': '...', 'conversation_id': null }")
    print("   Returns: { 'run_id': '...', 'status': 'started' }")
    print()
    print("2. GET /api/runs/{run_id}/events (SSE stream)")
    print("   Events:")
    print("   - run_started")
    print("   - decider_done")
    print("   - research_iteration_*")
    print("   - final_response (with evidence_pack)")
    print("   - run_completed")
    print()
    print("3. Evidence Pack Structure:")
    print("   {")
    print("     'sources': [...],")
    print("     'claims': [...],")
    print("     'conflicts': [...],")
    print("     'gaps': [...]")
    print("   }")
    print()
    print("=" * 60)
    print("Expected Results for Test Query")
    print("=" * 60)
    print()
    print("Query: 'Find recent news and regulatory information about Apple Inc.'")
    print()
    print("Expected Response:")
    print("- Executive summary with risk assessment")
    print("- Financial performance section")
    print("- Regulatory compliance status")
    print("- Recent news and events")
    print("- Evidence pack with:")
    print("  • Sources: SEC filings, Bloomberg, Reuters, WSJ")
    print("  • Claims: Extracted facts about Apple")
    print("  • Conflicts: Any contradictory information")
    print("  • Gaps: Missing information")
    print()
    print("Risk Level: Low/Medium/High")
    print("Sources: 6-20 authoritative sources")
    print("Citations: All sources with URLs")

if __name__ == "__main__":
    test_query()
    show_query_structure()

