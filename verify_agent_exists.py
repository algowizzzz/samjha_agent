#!/usr/bin/env python3
"""
Quick script to verify if Financial News Agent exists
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from external.core.db.session import get_db_session
    from external.agent.persistence import list_agents_db
    from external.agent.agent_registry import slugify_name
    
    agent_name = "Financial News Research Agent for Bankers"
    agent_id = slugify_name(agent_name)
    
    print(f"Checking for: {agent_name}")
    print(f"Agent ID: {agent_id}")
    print()
    
    with get_db_session() as db:
        agents = list_agents_db(db)
        
        # Find financial agent
        financial_agent = None
        for agent in agents:
            if agent.get('id') == agent_id:
                financial_agent = agent
                break
        
        if financial_agent:
            print("✅ AGENT EXISTS!")
            print(f"   ID: {financial_agent.get('id')}")
            print(f"   Name: {financial_agent.get('name')}")
            print(f"   Type: {financial_agent.get('agent_type')}")
            print(f"   Model: {financial_agent.get('model')}")
            print(f"\n   Chat URL: http://localhost:5000/agent/chat/{financial_agent.get('id')}")
        else:
            print("❌ AGENT NOT FOUND")
            print(f"\n   To create it:")
            print(f"   1. Via UI: http://localhost:5000/admin → Manage Agents → External → Create")
            print(f"   2. Via script: python3 setup_web_search_agent.py")
            
except ImportError as e:
    print("❌ Cannot check - missing dependencies")
    print(f"   Error: {e}")
    print(f"   Install: pip install sqlalchemy")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

