#!/usr/bin/env python3
"""
Test backend functions directly (database layer)
This tests the actual functionality without requiring authentication
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.db.session import get_db_session
from external.agent.persistence import (
    list_agents_db,
    get_agent_db,
    create_agent_db,
    list_agent_prompts,
    get_agent_prompt,
    list_prompts,
    get_prompt_content,
)

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(test_name, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"   {details}")

def test_list_agents():
    """Test listing agents from database"""
    print_section("1. List Agents (Database)")
    try:
        with get_db_session() as db:
            agents = list_agents_db(db)
            print_result("List agents", True, f"Found {len(agents)} agents")
            
            # Find financial news agent
            financial_agent = None
            for agent in agents:
                name = agent.get("name", "").lower()
                if "financial" in name and "banker" in name:
                    financial_agent = agent
                    break
            
            if financial_agent:
                print(f"   ✅ Found Financial News Agent:")
                print(f"      ID: {financial_agent.get('id')}")
                print(f"      Name: {financial_agent.get('name')}")
                print(f"      Type: {financial_agent.get('agent_type')}")
                return financial_agent.get('id')
            else:
                print(f"   ⚠️  Financial News Agent not found")
                print(f"   Available agents:")
                for agent in agents[:5]:  # Show first 5
                    print(f"      - {agent.get('name')} ({agent.get('agent_type')})")
                return None
    except Exception as e:
        print_result("List agents error", False, str(e))
        import traceback
        traceback.print_exc()
        return None

def test_get_agent(agent_id):
    """Test getting a specific agent"""
    print_section("2. Get Agent (Database)")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    try:
        with get_db_session() as db:
            agent = get_agent_db(db, agent_id)
            if agent:
                print_result("Get agent", True, f"Agent: {agent.get('name')}")
                print(f"   ID: {agent.get('id')}")
                print(f"   Type: {agent.get('agent_type')}")
                print(f"   Model: {agent.get('model')}")
                print(f"   Description: {agent.get('description', 'N/A')[:100]}...")
                print(f"   Research Depth: {agent.get('default_research_depth', 'N/A')}")
                
                allowed = agent.get('search_scope_allowed_domains', [])
                blocked = agent.get('search_scope_blocked_domains', [])
                print(f"   Allowed Domains: {len(allowed)} domains")
                if allowed:
                    print(f"      {', '.join(allowed[:5])}{'...' if len(allowed) > 5 else ''}")
                print(f"   Blocked Domains: {len(blocked)} domains")
                
                domain_content = agent.get('domain_content', '')
                if domain_content:
                    print(f"   Domain Content: {len(domain_content)} chars")
                
                return True
            else:
                print_result("Get agent", False, f"Agent not found: {agent_id}")
                return False
    except Exception as e:
        print_result("Get agent error", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_list_agent_prompts(agent_id):
    """Test listing agent prompts"""
    print_section("3. List Agent Prompts (Database)")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    try:
        with get_db_session() as db:
            prompts = list_agent_prompts(db, agent_id)
            print_result("List agent prompts", True, f"Found {len(prompts)} prompts")
            
            # Show prompt status
            overridden = sum(1 for p in prompts if p.get("is_overridden", False))
            defaults = len(prompts) - overridden
            print(f"   Overridden: {overridden}, Using defaults: {defaults}")
            
            # Show first few prompts
            print(f"\n   Prompt Details:")
            for prompt in prompts[:5]:
                status = "OVERRIDE" if prompt.get("is_overridden") else "DEFAULT"
                print(f"      [{status}] {prompt.get('name')} - {prompt.get('display_name', 'N/A')}")
            
            return True
    except Exception as e:
        print_result("List agent prompts error", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_get_agent_prompt(agent_id):
    """Test getting a specific agent prompt"""
    print_section("4. Get Agent Prompt (Database)")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    # Try to get the decider prompt
    prompt_name = "web_research_decider"
    try:
        with get_db_session() as db:
            agent_prompt = get_agent_prompt(db, agent_id, prompt_name)
            if agent_prompt:
                print_result("Get agent prompt (override)", True, f"Prompt: {prompt_name}")
                print(f"   Content length: {len(agent_prompt.content)} chars")
                print(f"   Is active: {agent_prompt.is_active}")
            else:
                # Try to get default prompt
                prompt_content = get_prompt_content(db, prompt_name, category="web_search", agent_id=agent_id)
                if prompt_content:
                    print_result("Get agent prompt (default)", True, f"Prompt: {prompt_name}")
                    print(f"   Content length: {len(prompt_content)} chars")
                    print(f"   Using default (no override)")
                else:
                    print_result("Get agent prompt", False, f"Prompt not found: {prompt_name}")
                    return False
            return True
    except Exception as e:
        print_result("Get agent prompt error", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_list_prompts():
    """Test listing system prompts"""
    print_section("5. List System Prompts (Database)")
    try:
        with get_db_session() as db:
            prompts = list_prompts(db, category="web_search")
            print_result("List system prompts", True, f"Found {len(prompts)} web_search prompts")
            
            print(f"\n   Prompt List:")
            for prompt in prompts:
                print(f"      - {prompt.get('name')} ({prompt.get('display_name', 'N/A')})")
            
            return True
    except Exception as e:
        print_result("List system prompts error", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_prompt_content():
    """Test getting prompt content"""
    print_section("6. Get Prompt Content (Database)")
    try:
        with get_db_session() as db:
            # Test getting a known prompt
            prompt_name = "web_research_decider"
            content = get_prompt_content(db, prompt_name, category="web_search")
            
            if content:
                print_result("Get prompt content", True, f"Prompt: {prompt_name}")
                print(f"   Content length: {len(content)} chars")
                print(f"   First 100 chars: {content[:100]}...")
                return True
            else:
                print_result("Get prompt content", False, f"Prompt not found: {prompt_name}")
                return False
    except Exception as e:
        print_result("Get prompt content error", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def test_agent_configuration(agent_id):
    """Test agent configuration completeness"""
    print_section("7. Agent Configuration Check")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    try:
        with get_db_session() as db:
            agent = get_agent_db(db, agent_id)
            if not agent:
                print_result("Agent configuration", False, "Agent not found")
                return False
            
            checks = {
                "Has ID": bool(agent.get('id')),
                "Has Name": bool(agent.get('name')),
                "Has Type": agent.get('agent_type') == 'external',
                "Has Model": bool(agent.get('model')),
                "Has Domain Content": bool(agent.get('domain_content')),
                "Has Allowed Domains": len(agent.get('search_scope_allowed_domains', [])) > 0,
                "Has Research Depth": bool(agent.get('default_research_depth')),
            }
            
            all_passed = all(checks.values())
            print_result("Agent configuration", all_passed, f"{sum(checks.values())}/{len(checks)} checks passed")
            
            print(f"\n   Configuration Details:")
            for check, passed in checks.items():
                status = "✅" if passed else "❌"
                print(f"      {status} {check}")
            
            return all_passed
    except Exception as e:
        print_result("Agent configuration error", False, str(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all backend function tests"""
    print("\n" + "=" * 60)
    print("  BACKEND FUNCTION TEST SUITE")
    print("  (Testing database layer directly)")
    print("=" * 60)
    
    results = {
        "list_agents": False,
        "get_agent": False,
        "list_agent_prompts": False,
        "get_agent_prompt": False,
        "list_prompts": False,
        "get_prompt_content": False,
        "agent_configuration": False,
    }
    
    agent_id = None
    
    # Test 1: List agents
    agent_id = test_list_agents()
    results["list_agents"] = agent_id is not None
    
    # Test 2: Get agent
    if agent_id:
        results["get_agent"] = test_get_agent(agent_id)
    
    # Test 3: List agent prompts
    if agent_id:
        results["list_agent_prompts"] = test_list_agent_prompts(agent_id)
    
    # Test 4: Get agent prompt
    if agent_id:
        results["get_agent_prompt"] = test_get_agent_prompt(agent_id)
    
    # Test 5: List system prompts
    results["list_prompts"] = test_list_prompts()
    
    # Test 6: Get prompt content
    results["get_prompt_content"] = test_prompt_content()
    
    # Test 7: Agent configuration
    if agent_id:
        results["agent_configuration"] = test_agent_configuration(agent_id)
    
    # Summary
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print()
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print()
    
    if passed_tests == total_tests:
        print("🎉 All backend functions work correctly!")
        print("\n✅ Backend APIs are ready to use")
        print("   (Note: API endpoints require authentication, but underlying functions work)")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️  Most tests passed. Some functions may need attention.")
    else:
        print("❌ Many tests failed. Check database connection and schema.")
    
    if agent_id:
        print(f"\nAgent ID: {agent_id}")
        print(f"To test in UI: http://localhost:5000/agent/chat/{agent_id}")

if __name__ == "__main__":
    main()

