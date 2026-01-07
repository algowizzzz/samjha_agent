"""Create a Financial News Quick Search Agent in the database"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from external.core.db.session import get_db_session
from external.agent.persistence import create_agent_db, get_agent_db
from external.core.db.models import Agent

AGENT_ID = "financial_news_quick_search"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def create_financial_news_agent():
    """Create the Financial News Quick Search agent"""
    if not TAVILY_API_KEY:
        print("⚠️  TAVILY_API_KEY not found in .env file")
        print("   Please set TAVILY_API_KEY in your .env file")
        return False
    
    with get_db_session() as db:
        # Check if agent already exists
        existing = get_agent_db(db, AGENT_ID)
        if existing:
            print(f"✓ Agent already exists: {AGENT_ID}")
            # Update Tavily API key if needed
            agent_obj = db.get(Agent, AGENT_ID)
            if agent_obj and not agent_obj.tavily_api_key:
                agent_obj.tavily_api_key = TAVILY_API_KEY
                db.commit()
                print(f"✓ Updated Tavily API key")
            return True
        
        # Create new agent
        agent = create_agent_db(
            db,
            agent_id=AGENT_ID,
            name="Financial News Quick Search",
            agent_type="quick_search",
            description="Quick search for financial news entities (sectors, counterparties, regulatory bodies). Searches Bloomberg, Reuters, WSJ, SEC, Federal Reserve, and other financial news sources.",
            tavily_api_key=TAVILY_API_KEY,
            quick_search_config={
                "max_results": 5,
                "search_depth": "basic",
                "include_answer": True,
                # Note: include_domains and exclude_domains are set in quick_search_agent.py
                # as FINANCIAL_NEWS_DOMAINS and BLOCKED_DOMAINS constants
            }
        )
        db.commit()
        print(f"✓ Created Financial News Quick Search agent: {agent.id}")
        print(f"  Name: {agent.name}")
        print(f"  Type: {agent.agent_type}")
        print(f"  Description: {agent.description}")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("Creating Financial News Quick Search Agent")
    print("=" * 60)
    
    if create_financial_news_agent():
        print("\n✅ Success! The agent should now appear in the Quick Search Agents section.")
        print("   You can test it with queries like:")
        print("   - 'RBC'")
        print("   - 'Apple stock price'")
        print("   - 'SEC regulatory updates'")
    else:
        print("\n❌ Failed to create agent")
    
    print("=" * 60)

