#!/usr/bin/env python3
"""Verify deep research integration is working"""
import os
from dotenv import load_dotenv

load_dotenv()

from external.core.db.session import get_db_session
from external.agent.persistence import list_prompts, list_agents_db

print("=" * 60)
print("Deep Research Integration Verification")
print("=" * 60)

# Check prompts
print("\n1. Checking Deep Research Prompts...")
with get_db_session() as db:
    prompts = list_prompts(db, category='deep_research')
    print(f"   Found {len(prompts)} prompts:")
    for p in prompts:
        print(f"   ✓ {p['display_name']} ({p['name']})")
    
    if len(prompts) == 0:
        print("   ✗ No prompts found! Run import_deep_research_prompts.py")
    elif len(prompts) < 7:
        print(f"   ⚠️  Expected 7 prompts, found {len(prompts)}")
    else:
        print("   ✅ All prompts imported successfully")

# Check agents
print("\n2. Checking Deep Research Agents...")
with get_db_session() as db:
    agents = list_agents_db(db)
    deep_research_agents = [a for a in agents if a.get('agent_type') == 'deep_research']
    print(f"   Found {len(deep_research_agents)} deep_research agents:")
    for a in deep_research_agents:
        print(f"   ✓ {a.get('name')} (ID: {a.get('id')})")
    
    if len(deep_research_agents) == 0:
        print("   ⚠️  No deep_research agents found. Create one in the admin panel.")
    else:
        print("   ✅ Deep research agents are available")

# Summary
print("\n" + "=" * 60)
print("Summary:")
print(f"  - Prompts: {len(prompts)}/7")
print(f"  - Agents: {len(deep_research_agents)}")
print("=" * 60)

