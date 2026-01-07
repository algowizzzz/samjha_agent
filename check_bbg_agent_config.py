"""Check BBG agent configuration in database"""
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "sqlite:///data/app.db")

from external.core.db.session import get_db_session
from external.agent.persistence import get_agent_db
import json

with get_db_session() as db:
    agent = get_agent_db(db, "bbg")
    if agent:
        print("Agent: BBG")
        print(f"  Name: {agent.get('name')}")
        print(f"  Type: {agent.get('agent_type')}")
        print(f"  Tavily API Key: {'Set' if agent.get('tavily_api_key') else 'Not set'}")
        print(f"  Quick Search Config: {json.dumps(agent.get('quick_search_config'), indent=2)}")
        
        # Also check raw DB record
        from external.core.db.models import Agent
        agent_obj = db.get(Agent, "bbg")
        if agent_obj:
            print(f"\nRaw DB quick_search_config: {agent_obj.quick_search_config}")
            print(f"Raw DB quick_search_config type: {type(agent_obj.quick_search_config)}")
    else:
        print("Agent 'bbg' not found")

