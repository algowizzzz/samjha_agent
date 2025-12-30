#!/usr/bin/env python3
"""
Test web search agent APIs with authentication.
"""

import requests
import json
import time
import re

BASE_URL = "http://localhost:8000"

# Default admin credentials
ADMIN_USER = "admin"
ADMIN_PASSWORD = "admin123"

def login():
    """Login and get session."""
    print("Logging in...")
    session = requests.Session()
    
    # First, get the login page to get any CSRF tokens or session cookies
    try:
        session.get(f"{BASE_URL}/login", timeout=5)
    except:
        pass
    
    # Try form login (this sets Flask session cookie)
    try:
        login_data = {
            'user_id': ADMIN_USER,
            'password': ADMIN_PASSWORD
        }
        response = session.post(
            f"{BASE_URL}/login",
            data=login_data,
            allow_redirects=True,
            timeout=10
        )
        
        # Check if we have a session cookie
        if 'session' in session.cookies or response.status_code in [200, 302]:
            # Check if redirected away from login
            final_url = response.url if hasattr(response, 'url') else response.request.url
            if 'login' not in final_url.lower() or response.status_code == 302:
                print("  ✓ Logged in successfully (session cookie set)")
                # Verify by accessing a protected page
                test_response = session.get(f"{BASE_URL}/admin", timeout=5, allow_redirects=False)
                if test_response.status_code != 302:  # Not redirected to login
                    print("  ✓ Session verified")
                    return session
                else:
                    print("  ⚠ Session may not be valid")
                    return session  # Return anyway, let's try
            else:
                print(f"  ✗ Still on login page")
                print(f"    Final URL: {final_url}")
                return None
        else:
            print(f"  ✗ No session cookie set")
            print(f"    Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ✗ Login error: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_prompts(session):
    """Check if web search prompts are imported."""
    print("\nChecking web search prompts...")
    try:
        response = session.get(f"{BASE_URL}/api/admin/prompts?category=web_search", timeout=10)
        print(f"  Status: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Response length: {len(response.text)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                prompts = data.get("prompts", [])
                print(f"  Found {len(prompts)} web search prompts")
                for prompt in prompts:
                    print(f"    - {prompt.get('name')}: {prompt.get('display_name')}")
                return len(prompts) >= 6
            except json.JSONDecodeError:
                print(f"  Response is not JSON: {response.text[:300]}")
                return False
        else:
            print(f"  Failed: {response.status_code}")
            print(f"    Response: {response.text[:300]}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_agents(session):
    """List all agents."""
    print("\nListing agents...")
    try:
        response = session.get(f"{BASE_URL}/api/admin/agents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            agents = data.get("agents", [])
            print(f"  Found {len(agents)} total agents")
            
            external_agents = [a for a in agents if a.get("agent_type") == "external"]
            print(f"  External agents: {len(external_agents)}")
            for agent in external_agents:
                print(f"    - {agent.get('name')} ({agent.get('id')})")
            
            return agents
        else:
            print(f"  Failed: {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return []
    except Exception as e:
        print(f"  Error: {e}")
        return []

def create_financial_news_agent(session):
    """Create financial news agent."""
    print("\nCreating financial news agent...")
    
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
    
    files = {
        'domain_file': ('financial_news_domain.md', domain_content, 'text/markdown')
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
        response = session.post(
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
            print(f"    Type: {agent.get('agent_type')}")
            return agent_id
        elif response.status_code == 409:
            # Agent already exists
            result = response.json()
            error = result.get("error", "")
            if "already exists" in error:
                # Extract agent_id from error or try to find it
                print(f"  Agent already exists, finding it...")
                agents = list_agents(session)
                for agent in agents:
                    if agent.get("name") == "Financial News Research Agent":
                        agent_id = agent.get("id")
                        print(f"  ✓ Found existing agent: {agent_id}")
                        return agent_id
            print(f"  ✗ Conflict: {error}")
            return None
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"    Response: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agent_run(session, agent_id, query="What are the latest trends in financial markets in 2024?"):
    """Test running an agent query."""
    print(f"\nTesting agent run for {agent_id}...")
    print(f"  Query: {query}")
    
    try:
        response = session.post(
            f"{BASE_URL}/api/agents/{agent_id}/runs",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            run_id = data.get("run_id")
            print(f"  ✓ Run created: {run_id}")
            
            # Wait a bit
            print("  Waiting for processing...")
            time.sleep(3)
            
            # Get run events
            try:
                events_response = session.get(
                    f"{BASE_URL}/api/runs/{run_id}/events",
                    timeout=10
                )
                if events_response.status_code == 200:
                    events_data = events_response.json()
                    events = events_data.get("events", [])
                    print(f"  Found {len(events)} events")
                    for event in events[-10:]:  # Show last 10 events
                        event_type = event.get("event_type", "unknown")
                        payload = event.get("payload", {})
                        print(f"    - {event_type}")
                        if event_type == "final_response":
                            response_text = payload.get("response", "")[:200]
                            print(f"      Response preview: {response_text}...")
                    return run_id
            except Exception as e:
                print(f"  ⚠ Could not get events: {e}")
            
            return run_id
        else:
            print(f"  ✗ Failed: {response.status_code}")
            print(f"    Response: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("=" * 60)
    print("Web Search Agent API Testing (with Authentication)")
    print("=" * 60)
    
    # Login
    session = login()
    if not session:
        print("\n✗ Cannot proceed without authentication")
        return
    
    # Check prompts
    prompts_ok = check_prompts(session)
    if not prompts_ok:
        print("\n⚠ Web search prompts may not be imported yet.")
        print("  They should be imported automatically on server start.")
    
    # List agents
    agents = list_agents(session)
    
    # Find or create financial news agent
    financial_agent = None
    for agent in agents:
        if agent.get("name") == "Financial News Research Agent":
            financial_agent = agent.get("id")
            print(f"\nFound existing financial news agent: {financial_agent}")
            break
    
    if not financial_agent:
        financial_agent = create_financial_news_agent(session)
    
    if financial_agent:
        # Test agent run
        print("\n" + "=" * 60)
        print("Testing Agent Run")
        print("=" * 60)
        run_id = test_agent_run(
            session,
            financial_agent,
            "What are the latest trends in financial markets in 2024?"
        )
        
        if run_id:
            print(f"\n✓ Test run completed: {run_id}")
            print(f"  Check run details at: {BASE_URL}/api/runs/{run_id}")
        else:
            print("\n⚠ Could not create test run")
    else:
        print("\n⚠ Could not create/find financial news agent")
    
    print("\n" + "=" * 60)
    print("Testing Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()

