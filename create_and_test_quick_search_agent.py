"""Create a quick search agent and test it with 'RBC' query"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure DATABASE_URL is set
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "sqlite:///data/app.db")

from external.core.db.session import get_db_session
from external.agent.persistence import create_agent_db, get_agent_db, update_agent_db
from external.agent.quick_search_agent import handle_quick_search_query

AGENT_ID = "financial_news_quick_search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def create_agent():
    """Create or update the quick search agent"""
    from external.core.db.models import Agent
    
    with get_db_session() as db:
        existing_agent = db.get(Agent, AGENT_ID)
        if existing_agent:
            print(f"✓ Agent already exists: {AGENT_ID}")
            # Update with Tavily API key if not set
            if not existing_agent.tavily_api_key and TAVILY_API_KEY:
                existing_agent.tavily_api_key = TAVILY_API_KEY
                existing_agent.name = "Financial News Quick Search"
                existing_agent.description = "Quick search for financial news entities (sectors, counterparties, regulatory bodies)"
                existing_agent.agent_type = "quick_search"
                if hasattr(existing_agent, 'quick_search_config'):
                    existing_agent.quick_search_config = {
                        "max_results": 5,
                        "search_depth": "basic",
                        "include_answer": True,
                    }
                db.commit()
                print(f"✓ Updated agent with Tavily API key and config")
            return AGENT_ID
        
        # Create new agent
        agent = create_agent_db(
            db,
            id=AGENT_ID,
            name="Financial News Quick Search",
            agent_type="quick_search",
            description="Quick search for financial news entities (sectors, counterparties, regulatory bodies)",
            tavily_api_key=TAVILY_API_KEY,
            quick_search_config={
                "max_results": 5,
                "search_depth": "basic",
                "include_answer": True,
            }
        )
        db.commit()
        print(f"✓ Created agent: {agent.id}")
        return agent.id

def test_query(agent_id):
    """Test the quick search agent with 'RBC' query"""
    print(f"\n🔍 Testing query: 'RBC'")
    print(f"🤖 Agent ID: {agent_id}\n")
    
    result = handle_quick_search_query(
        user_query="RBC",
        conversation_history=[],
        prior_state=None,
        tools_registry=None,
        policy_limits=None,
        show_thinking=False,
        agent_id=agent_id,
        on_decider_output=None,
        continuity_packet=None,
    )
    
    status = result.get("status")
    print(f"\n📊 Status: {status}")
    
    if status == "SUCCESS":
        final_answer = result.get("final_answer", "")
        sources = result.get("sources", [])
        
        print(f"\n✅ Response:\n")
        print(final_answer)
        print(f"\n📚 Sources found: {len(sources)}")
        if sources:
            print("\nTop sources:")
            for i, source in enumerate(sources[:3], 1):
                print(f"  {i}. {source.get('title', 'N/A')}")
                print(f"     {source.get('url', 'N/A')}")
    else:
        error = result.get("last_error", "Unknown error")
        print(f"\n❌ Error: {error}")
        print(f"Error type: {result.get('error_type', 'N/A')}")

if __name__ == "__main__":
    if not TAVILY_API_KEY:
        print("⚠️  TAVILY_API_KEY not found in .env file")
        print("   Please set TAVILY_API_KEY in your .env file")
        exit(1)
    
    print("=" * 60)
    print("Quick Search Agent Test - RBC Query")
    print("=" * 60)
    
    agent_id = create_agent()
    test_query(agent_id)
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

