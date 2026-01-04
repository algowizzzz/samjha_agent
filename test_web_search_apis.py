#!/usr/bin/env python3
"""
Test web search agent APIs:
1. Import prompts (via admin API if available, or direct DB)
2. Create financial news agent
3. List agents
4. Test agent run
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_server_health():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def import_prompts_via_api():
    """Import prompts - this happens automatically on server start, but we can verify."""
    print("Checking prompts...")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/prompts?category=web_search")
        if response.status_code == 200:
            data = response.json()
            prompts = data.get("prompts", [])
            print(f"  Found {len(prompts)} web search prompts")
            for prompt in prompts:
                print(f"    - {prompt.get('name')}: {prompt.get('display_name')}")
            return len(prompts) > 0
        else:
            print(f"  Failed to get prompts: {response.status_code}")
            return False
    except Exception as e:
        print(f"  Error checking prompts: {e}")
        return False

def create_financial_news_agent():
    """Create financial news agent via API."""
    print("\nCreating financial news agent via API...")
    
    # Domain content for financial news
    domain_content = """# Financial News Research Agent Domain Configuration

## 1. Domain Identity
- **Domain Key**: financial_news
- **Purpose**: Research and analyze financial news, market trends, and economic indicators

## 2. Authority Domains
Primary authoritative sources for financial information:
- **sec.gov** - Securities and Exchange Commission
- **federalreserve.gov** - Federal Reserve
- **treasury.gov** - U.S. Treasury
- **bloomberg.com** - Bloomberg
- **reuters.com** - Reuters
- **wsj.com** - Wall Street Journal
- **ft.com** - Financial Times

## 3. Research Depth Settings
- **Default Research Depth**: Standard
- **Max Iterations**: 2-3
- **Min Sources**: 6
- **Max Sources**: 20

## 4. Search Scope
- **Allowed Domains**: sec.gov, federalreserve.gov, treasury.gov, bloomberg.com, reuters.com, wsj.com, ft.com
- **Blocked Domains**: reddit.com, twitter.com, facebook.com
"""
    
    # Create domain file content
    domain_file_content = domain_content
    
    # Prepare form data
    files = {
        'domain_file': ('financial_news_domain.md', domain_file_content, 'text/markdown')
    }
    
    data = {
        'name': 'Financial News Research Agent',
        'description': 'Web research agent focused on financial news, market trends, and economic indicators',
        'agent_type': 'external',
        'model': 'claude-3-sonnet-20240229',
        'allowed_domains': 'sec.gov,federalreserve.gov,treasury.gov,bloomberg.com,reuters.com,wsj.com,ft.com',
        'blocked_domains': 'reddit.com,twitter.com,facebook.com',
        'default_research_depth': 'standard'
    }
    
    try:
        # Note: This requires admin authentication
        # For testing, we'll try without auth first
        response = requests.post(
            f"{BASE_URL}/api/admin/agents",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            agent = result.get("agent", {})
            agent_id = agent.get("id")
            print(f"  ✓ Agent created: {agent_id}")
            print(f"    Name: {agent.get('name')}")
            return agent_id
        elif response.status_code == 401:
            print(f"  ⚠ Authentication required. Please login first.")
            print(f"    Response: {response.text}")
            return None
        else:
            print(f"  ✗ Failed to create agent: {response.status_code}")
            print(f"    Response: {response.text}")
            return None
    except Exception as e:
        print(f"  ✗ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def list_agents():
    """List all agents."""
    print("\nListing agents...")
    try:
        response = requests.get(f"{BASE_URL}/api/admin/agents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            agents = data.get("agents", [])
            print(f"  Found {len(agents)} agents")
            
            external_agents = [a for a in agents if a.get("agent_type") == "external"]
            print(f"  External agents: {len(external_agents)}")
            for agent in external_agents:
                print(f"    - {agent.get('name')} ({agent.get('id')})")
            
            return agents
        else:
            print(f"  Failed to list agents: {response.status_code}")
            return []
    except Exception as e:
        print(f"  Error listing agents: {e}")
        return []

def test_agent_run(agent_id, query="What are the latest trends in financial markets?"):
    """Test running an agent query."""
    print(f"\nTesting agent run for {agent_id}...")
    print(f"  Query: {query}")
    
    try:
        # Create a run
        response = requests.post(
            f"{BASE_URL}/api/agents/{agent_id}/runs",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            run_id = data.get("run_id")
            print(f"  ✓ Run created: {run_id}")
            
            # Wait a bit for processing
            print("  Waiting for run to process...")
            time.sleep(2)
            
            # Get run events (SSE endpoint)
            try:
                events_response = requests.get(
                    f"{BASE_URL}/api/runs/{run_id}/events",
                    timeout=5,
                    stream=False
                )
                if events_response.status_code == 200:
                    events_data = events_response.json()
                    events = events_data.get("events", [])
                    print(f"  Found {len(events)} events")
                    for event in events[-5:]:  # Show last 5 events
                        event_type = event.get("event_type", "unknown")
                        print(f"    - {event_type}")
                    return run_id
            except Exception as e:
                print(f"  ⚠ Could not get events: {e}")
            
            return run_id
        else:
            print(f"  ✗ Failed to create run: {response.status_code}")
            print(f"    Response: {response.text}")
            return None
    except Exception as e:
        print(f"  ✗ Error running agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("=" * 60)
    print("Web Search Agent API Testing")
    print("=" * 60)
    
    # Check server
    print("\n1. Checking server health...")
    if not test_server_health():
        print("  ✗ Server is not running!")
        print("  Please start the server first: python3 run_server.py")
        return
    print("  ✓ Server is running")
    
    # Check prompts
    print("\n2. Checking web search prompts...")
    prompts_ok = import_prompts_via_api()
    if not prompts_ok:
        print("  ⚠ Web search prompts not found. They should be imported on server start.")
    
    # List existing agents
    print("\n3. Listing existing agents...")
    agents = list_agents()
    
    # Find or create financial news agent
    financial_agent = None
    for agent in agents:
        if agent.get("name") == "Financial News Research Agent":
            financial_agent = agent.get("id")
            print(f"\n4. Found existing financial news agent: {financial_agent}")
            break
    
    if not financial_agent:
        print("\n4. Creating financial news agent...")
        financial_agent = create_financial_news_agent()
    
    if financial_agent:
        # Test agent run
        print("\n5. Testing agent run...")
        run_id = test_agent_run(financial_agent, "What are the latest trends in financial markets in 2024?")
        
        if run_id:
            print(f"\n  ✓ Test run created: {run_id}")
            print(f"  You can check the run status at: {BASE_URL}/api/runs/{run_id}")
        else:
            print("\n  ⚠ Could not create test run (may require authentication)")
    else:
        print("\n  ⚠ Could not create/find financial news agent")
    
    print("\n" + "=" * 60)
    print("API Testing Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()


