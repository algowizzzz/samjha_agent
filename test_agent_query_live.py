#!/usr/bin/env python3
"""
Test agent query with actual API calls
"""

import sys
import os
from pathlib import Path

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from core.db.session import get_db_session
from external.agent.persistence import get_agent_db
from external.agent.web_research_agent import handle_web_research_query

def test_query():
    """Test a real query"""
    
    agent_id = "financial_news_research_agent_for_bankers"
    test_query = "Find recent news and regulatory information about Apple Inc. for a credit risk assessment"
    
    print("=" * 60)
    print("Testing Financial News Agent Query (LIVE)")
    print("=" * 60)
    print(f"Agent ID: {agent_id}")
    print(f"Query: {test_query}")
    print()
    
    # Verify agent exists
    with get_db_session() as db:
        agent = get_agent_db(db, agent_id)
        if not agent:
            print("❌ Agent not found!")
            return
        print(f"✅ Agent found: {agent.get('name')}")
        print()
    
    print("Executing query...")
    print("(This will make real API calls to Tavily and LLM)")
    print("(May take 1-3 minutes)")
    print()
    print("-" * 60)
    print()
    
    try:
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
            print(response_text)
            print()
            print("-" * 60)
            print()
            
            if evidence_pack:
                sources = evidence_pack.get("sources", [])
                claims = evidence_pack.get("claims", [])
                conflicts = evidence_pack.get("conflicts", [])
                gaps = evidence_pack.get("gaps", [])
                
                print("Evidence Pack:")
                print(f"  📚 Sources: {len(sources)}")
                if sources:
                    print()
                    print("  Sample Sources:")
                    for i, source in enumerate(sources[:5], 1):
                        title = source.get('title', source.get('url', 'N/A'))
                        url = source.get('url', 'N/A')
                        print(f"    {i}. {title}")
                        print(f"       {url}")
                
                print()
                print(f"  ✅ Claims: {len(claims)}")
                if claims:
                    print()
                    print("  Sample Claims:")
                    for i, claim in enumerate(claims[:3], 1):
                        text = claim.get('claim_text', claim.get('text', 'N/A'))
                        confidence = claim.get('confidence', 'N/A')
                        print(f"    {i}. [{confidence}] {text[:100]}...")
                
                print()
                print(f"  ⚠️  Conflicts: {len(conflicts)}")
                if conflicts:
                    for i, conflict in enumerate(conflicts[:2], 1):
                        claim1 = conflict.get('claim1', 'N/A')
                        claim2 = conflict.get('claim2', 'N/A')
                        print(f"    {i}. {claim1[:80]}... vs {claim2[:80]}...")
                
                print()
                print(f"  ❓ Gaps: {len(gaps)}")
                if gaps:
                    for i, gap in enumerate(gaps[:2], 1):
                        desc = gap.get('gap_description', gap.get('description', 'N/A'))
                        print(f"    {i}. {desc}")
        
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
            import traceback
            if "traceback" in result:
                print(result["traceback"])
        
        print()
        print("=" * 60)
        print("Test Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query()

