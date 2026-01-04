#!/usr/bin/env python3
"""
Test script for backend APIs
Tests agent creation, retrieval, prompts, and query execution
"""

import sys
import json
import requests
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5000"
TIMEOUT = 30

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_result(test_name, success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"   {details}")

def test_health_check():
    """Test if Flask app is running"""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        # 200, 302 (redirect), 401 (auth required), 403 (forbidden) all indicate app is running
        if response.status_code in [200, 302, 401, 403]:
            print_result("Flask app is running", True, f"Status: {response.status_code} (auth may be required)")
            return True
        else:
            print_result("Flask app response", False, f"Status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_result("Flask app connection", False, "App is not running. Start with: python app.py")
        return False
    except Exception as e:
        print_result("Health check error", False, str(e))
        return False

def test_list_agents():
    """Test listing agents"""
    print_section("2. List Agents API")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/agents", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            agents = data.get("agents", [])
            print_result("List agents", True, f"Found {len(agents)} agents")
            
            # Find financial news agent
            financial_agent = None
            for agent in agents:
                if "financial" in agent.get("name", "").lower() and "banker" in agent.get("name", "").lower():
                    financial_agent = agent
                    break
            
            if financial_agent:
                print(f"   ✅ Found Financial News Agent: {financial_agent.get('id')}")
                return financial_agent.get('id')
            else:
                print(f"   ⚠️  Financial News Agent not found in list")
                return None
        elif response.status_code in [401, 403]:
            print_result("List agents", False, f"Authentication required ({response.status_code})")
            print("   ⚠️  Note: Admin endpoints require authentication")
            print("   This is expected behavior - APIs are protected")
            return None
        else:
            print_result("List agents", False, f"Status: {response.status_code}, Response: {response.text[:200]}")
            return None
    except Exception as e:
        print_result("List agents error", False, str(e))
        return None

def test_get_agent(agent_id):
    """Test getting a specific agent"""
    print_section("3. Get Agent API")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    try:
        response = requests.get(f"{BASE_URL}/api/admin/agents/{agent_id}", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            agent = data.get("agent", {})
            print_result("Get agent", True, f"Agent: {agent.get('name')}")
            print(f"   Type: {agent.get('agent_type')}")
            print(f"   Model: {agent.get('model')}")
            print(f"   Research Depth: {agent.get('default_research_depth', 'N/A')}")
            print(f"   Allowed Domains: {len(agent.get('search_scope_allowed_domains', []))} domains")
            return True
        elif response.status_code in [401, 403]:
            print_result("Get agent", False, f"Authentication required ({response.status_code})")
            return False
        elif response.status_code == 404:
            print_result("Get agent", False, f"Agent not found: {agent_id}")
            return False
        else:
            print_result("Get agent", False, f"Status: {response.status_code}, Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_result("Get agent error", False, str(e))
        return False

def test_list_agent_prompts(agent_id):
    """Test listing agent prompts"""
    print_section("4. List Agent Prompts API")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    try:
        response = requests.get(f"{BASE_URL}/api/admin/agents/{agent_id}/prompts", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            prompts = data.get("prompts", [])
            print_result("List agent prompts", True, f"Found {len(prompts)} prompts")
            
            # Show prompt status
            overridden = sum(1 for p in prompts if p.get("is_overridden", False))
            defaults = len(prompts) - overridden
            print(f"   Overridden: {overridden}, Using defaults: {defaults}")
            
            return True
        elif response.status_code in [401, 403]:
            print_result("List agent prompts", False, f"Authentication required ({response.status_code})")
            return False
        else:
            print_result("List agent prompts", False, f"Status: {response.status_code}, Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_result("List agent prompts error", False, str(e))
        return False

def test_get_agent_prompt(agent_id):
    """Test getting a specific agent prompt"""
    print_section("5. Get Agent Prompt API")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    # Try to get the decider prompt
    prompt_name = "web_research_decider"
    try:
        response = requests.get(f"{BASE_URL}/api/admin/agents/{agent_id}/prompts/{prompt_name}", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print_result("Get agent prompt", True, f"Prompt: {prompt_name}")
            print(f"   Is default: {data.get('is_default', False)}")
            print(f"   Content length: {len(data.get('content', ''))} chars")
            return True
        elif response.status_code in [401, 403]:
            print_result("Get agent prompt", False, f"Authentication required ({response.status_code})")
            return False
        elif response.status_code == 404:
            print_result("Get agent prompt", False, f"Prompt not found: {prompt_name}")
            return False
        else:
            print_result("Get agent prompt", False, f"Status: {response.status_code}, Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_result("Get agent prompt error", False, str(e))
        return False

def test_list_prompts():
    """Test listing system prompts"""
    print_section("6. List System Prompts API")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/prompts?category=web_search", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            prompts = data.get("prompts", [])
            print_result("List system prompts", True, f"Found {len(prompts)} web_search prompts")
            return True
        elif response.status_code in [401, 403]:
            print_result("List system prompts", False, f"Authentication required ({response.status_code})")
            return False
        else:
            print_result("List system prompts", False, f"Status: {response.status_code}, Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_result("List system prompts error", False, str(e))
        return False

def test_create_agent_run(agent_id):
    """Test creating an agent run (query)"""
    print_section("7. Create Agent Run API")
    if not agent_id:
        print("   ⚠️  Skipping - no agent ID available")
        return False
    
    # Test query
    test_query = "Find recent news about Apple Inc. for credit risk assessment"
    
    try:
        payload = {
            "query": test_query,
            "conversation_id": None  # New conversation
        }
        
        print(f"   Query: {test_query}")
        print(f"   Starting run... (this may take time)")
        
        response = requests.post(
            f"{BASE_URL}/api/agents/{agent_id}/runs",
            json=payload,
            timeout=60,  # Longer timeout for actual query
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            run_id = data.get("run_id")
            print_result("Create agent run", True, f"Run ID: {run_id}")
            print(f"   Status: {data.get('status', 'unknown')}")
            return run_id
        elif response.status_code in [401, 403]:
            print_result("Create agent run", False, f"Authentication required ({response.status_code})")
            return None
        else:
            print_result("Create agent run", False, f"Status: {response.status_code}, Response: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print_result("Create agent run", False, "Request timed out (query may be processing)")
        return None
    except Exception as e:
        print_result("Create agent run error", False, str(e))
        return None

def test_get_run_events(run_id):
    """Test getting run events (SSE endpoint)"""
    print_section("8. Get Run Events API")
    if not run_id:
        print("   ⚠️  Skipping - no run ID available")
        return False
    
    try:
        # Note: This is an SSE endpoint, so we'll just check if it's accessible
        response = requests.get(
            f"{BASE_URL}/api/runs/{run_id}/events",
            timeout=5,
            headers={"Accept": "text/event-stream"}
        )
        
        if response.status_code == 200:
            print_result("Get run events", True, "SSE endpoint accessible")
            # Try to read a bit of the stream
            content = response.text[:500] if hasattr(response, 'text') else ""
            if content:
                print(f"   Stream preview: {content[:100]}...")
            return True
        elif response.status_code in [401, 403]:
            print_result("Get run events", False, f"Authentication required ({response.status_code})")
            return False
        else:
            print_result("Get run events", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_result("Get run events error", False, str(e))
        return False

def main():
    """Run all API tests"""
    print("\n" + "=" * 60)
    print("  BACKEND API TEST SUITE")
    print("=" * 60)
    
    results = {
        "health_check": False,
        "list_agents": False,
        "get_agent": False,
        "list_agent_prompts": False,
        "get_agent_prompt": False,
        "list_prompts": False,
        "create_run": False,
        "get_events": False
    }
    
    agent_id = None
    run_id = None
    
    # Test 1: Health check
    results["health_check"] = test_health_check()
    if not results["health_check"]:
        print("\n❌ Flask app is not running. Please start it with: python app.py")
        return
    
    # Test 2: List agents
    agent_id = test_list_agents()
    results["list_agents"] = agent_id is not None
    
    # Test 3: Get agent
    if agent_id:
        results["get_agent"] = test_get_agent(agent_id)
    
    # Test 4: List agent prompts
    if agent_id:
        results["list_agent_prompts"] = test_list_agent_prompts(agent_id)
    
    # Test 5: Get agent prompt
    if agent_id:
        results["get_agent_prompt"] = test_get_agent_prompt(agent_id)
    
    # Test 6: List system prompts
    results["list_prompts"] = test_list_prompts()
    
    # Test 7: Create agent run (optional - may require API keys)
    if agent_id:
        print("\n   ⚠️  Note: Agent run test requires Tavily and LLM API keys")
        print("   This test will make actual API calls and may incur costs")
        response = input("   Run agent query test? (y/n): ").strip().lower()
        if response == 'y':
            run_id = test_create_agent_run(agent_id)
            results["create_run"] = run_id is not None
            
            # Test 8: Get run events
            if run_id:
                results["get_events"] = test_get_run_events(run_id)
    
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
        print("🎉 All tests passed!")
    elif passed_tests >= total_tests * 0.7:
        print("⚠️  Most tests passed. Some may require authentication or API keys.")
    else:
        print("❌ Many tests failed. Check Flask app status and authentication.")
    
    if agent_id:
        print(f"\nAgent ID for manual testing: {agent_id}")
        print(f"Chat URL: {BASE_URL}/agent/chat/{agent_id}")

if __name__ == "__main__":
    main()

