#!/usr/bin/env python3
"""
Test agent query directly (bypassing API, calling handler function)
This shows what the actual query execution would produce
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.db.session import get_db_session
from external.agent.persistence import get_agent_db
from external.agent.web_research_agent import handle_web_research_query, initialize_research_controller_state

def test_query_direct():
    """Test query by calling handler directly"""
    
    agent_id = "financial_news_research_agent_for_bankers"
    test_query = "Find recent news and regulatory information about Apple Inc. for a credit risk assessment"
    
    print("=" * 60)
    print("Testing Financial News Agent Query (Direct)")
    print("=" * 60)
    print(f"Agent ID: {agent_id}")
    print(f"Query: {test_query}")
    print()
    
    # Get agent
    with get_db_session() as db:
        agent = get_agent_db(db, agent_id)
        if not agent:
            print("❌ Agent not found!")
            return
        
        print(f"✅ Agent found: {agent.get('name')}")
        print(f"   Type: {agent.get('agent_type')}")
        print(f"   Model: {agent.get('model')}")
        print()
    
    print("⚠️  Note: This will make actual API calls to:")
    print("   - Tavily API (for web search)")
    print("   - LLM API (for processing)")
    print("   This requires API keys to be configured.")
    print()
    
    response = input("Do you want to proceed? This will incur API costs. (y/n): ").strip().lower()
    if response != 'y':
        print("Skipping actual execution.")
        print()
        print("To test with actual execution:")
        print("1. Ensure Tavily API key is set in environment or agent config")
        print("2. Ensure LLM API key is set in environment")
        print("3. Run this script again and answer 'y'")
        print()
        show_expected_results()
        return
    
    print()
    print("Initializing controller state...")
    
    try:
        # Initialize state
        initial_state = initialize_research_controller_state(
            user_query=test_query,
            conversation_history=[],
            agent_id=agent_id,
            policy_limits={
                "max_iterations": 2,
                "max_sources": 20,
                "cost_limit": None
            }
        )
        
        print("✅ State initialized")
        print()
        print("Executing query...")
        print("(This may take 1-3 minutes)")
        print()
        
        # Execute query
        result = handle_web_research_query(
            user_query=test_query,
            conversation_history=[],
            agent_id=agent_id,
            show_thinking=False
        )
        
        print()
        print("=" * 60)
        print("QUERY RESULTS")
        print("=" * 60)
        print()
        
        # Display results
        status = result.get("status")
        print(f"Status: {status}")
        print()
        
        if status == "SUCCESS":
            response_text = result.get("response", "")
            evidence_pack = result.get("evidence_pack", {})
            
            print("Response:")
            print("-" * 60)
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            print()
            
            if evidence_pack:
                sources = evidence_pack.get("sources", [])
                claims = evidence_pack.get("claims", [])
                conflicts = evidence_pack.get("conflicts", [])
                gaps = evidence_pack.get("gaps", [])
                
                print("Evidence Pack:")
                print(f"  Sources: {len(sources)}")
                print(f"  Claims: {len(claims)}")
                print(f"  Conflicts: {len(conflicts)}")
                print(f"  Gaps: {len(gaps)}")
                print()
                
                if sources:
                    print("Sample Sources:")
                    for i, source in enumerate(sources[:3], 1):
                        print(f"  {i}. {source.get('title', source.get('url', 'N/A'))}")
                        print(f"     URL: {source.get('url', 'N/A')}")
        
        elif status == "ASK_USER":
            question = result.get("question", "")
            print("Agent needs clarification:")
            print(f"  {question}")
        
        elif status == "BLOCK":
            reason = result.get("reason", "")
            print("Query was blocked:")
            print(f"  {reason}")
        
        elif status == "ERROR":
            error = result.get("error", "")
            print("Error occurred:")
            print(f"  {error}")
        
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        import traceback
        traceback.print_exc()
        print()
        show_expected_results()

def show_expected_results():
    """Show what the expected results would look like"""
    print("=" * 60)
    print("Expected Query Results")
    print("=" * 60)
    print()
    print("Query: 'Find recent news and regulatory information about Apple Inc.'")
    print()
    print("Expected Response Format:")
    print()
    print("# Company Research Report: Apple Inc. (AAPL)")
    print()
    print("**Research Date:** [Current Date]")
    print("**Risk Level:** Low")
    print("**Sources:** 12 authoritative sources")
    print()
    print("## Executive Summary")
    print()
    print("Apple Inc. (AAPL) demonstrates strong financial performance...")
    print()
    print("## Financial Performance")
    print()
    print("**Key Metrics (from SEC Form 10-K):**")
    print("- Revenue: $[amount]")
    print("- Net Income: $[amount]")
    print()
    print("## Regulatory Compliance")
    print()
    print("**SEC Filings Status:**")
    print("- ✅ All required filings submitted on time")
    print("- ✅ No enforcement actions")
    print()
    print("## Evidence Pack")
    print()
    print("**Sources (12):**")
    print("1. SEC Form 10-K - Apple Inc. (filed [date])")
    print("2. Bloomberg: '[Article Title]' - Published [date]")
    print("3. Reuters: '[Article Title]' - Published [date]")
    print("...")
    print()
    print("**Claims (15):**")
    print("- Apple reported revenue of $X in Q3 2024")
    print("- Apple maintains strong cash position")
    print("...")
    print()
    print("**Conflicts (0):**")
    print("No conflicting information found")
    print()
    print("**Gaps (2):**")
    print("- Limited information on recent M&A activity")
    print("...")

if __name__ == "__main__":
    test_query_direct()

