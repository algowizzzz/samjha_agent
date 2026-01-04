#!/usr/bin/env python3
"""
Test agent query with QUICK research depth (fastest option)
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from external.agent.web_research_agent import handle_web_research_query

def test_quick_query():
    """Test with quick research depth"""
    
    agent_id = "financial_news_research_agent_for_bankers"
    test_query = "What is Apple Inc?"
    
    print("=" * 60)
    print("QUICK QUERY TEST (Fastest Research Depth)")
    print("=" * 60)
    print(f"Query: {test_query}")
    print()
    print("Using QUICK research depth:")
    print("  - 1 iteration only")
    print("  - 3-10 sources")
    print("  - Fastest response")
    print()
    print("Executing...")
    print("-" * 60)
    print()
    
    # Override policy limits for quick research
    quick_policy_limits = {
        "max_attempts": 2,
        "max_iterations": 1,  # Only 1 iteration for quick
        "max_sources": 10,     # Limit to 10 sources
        "max_api_calls": 5,   # Limit API calls
        "timeout_seconds": 30,
    }
    
    try:
        result = handle_web_research_query(
            user_query=test_query,
            conversation_history=[],
            agent_id=agent_id,
            show_thinking=False,
            policy_limits=quick_policy_limits  # Override for quick research
        )
        
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print()
        
        status = result.get("status")
        print(f"Status: {status}")
        print()
        
        if status == "SUCCESS":
            response = result.get("response", "")
            evidence = result.get("evidence_pack", {})
            
            print(f"Response Length: {len(response)} characters")
            print()
            
            if response:
                print("RESPONSE:")
                print("-" * 60)
                print(response)
                print("-" * 60)
            else:
                print("⚠️  Response text is empty")
            
            print()
            print("EVIDENCE PACK:")
            print("-" * 60)
            
            sources = evidence.get("sources", [])
            claims = evidence.get("claims", [])
            conflicts = evidence.get("conflicts", [])
            gaps = evidence.get("gaps", [])
            
            print(f"📚 Sources: {len(sources)}")
            if sources:
                print()
                for i, src in enumerate(sources[:5], 1):
                    title = src.get("title", src.get("url", "N/A"))
                    url = src.get("url", "N/A")
                    print(f"  {i}. {title}")
                    print(f"     {url}")
            
            print()
            print(f"✅ Claims: {len(claims)}")
            if claims:
                print()
                for i, claim in enumerate(claims[:3], 1):
                    text = claim.get("claim_text", claim.get("text", "N/A"))
                    conf = claim.get("confidence", "N/A")
                    print(f"  {i}. [{conf}] {text[:150]}...")
            
            print()
            print(f"⚠️  Conflicts: {len(conflicts)}")
            print(f"❓ Gaps: {len(gaps)}")
            
        elif status == "ERROR":
            print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"Result: {result}")
        
        print()
        print("=" * 60)
        print("Test Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quick_query()

