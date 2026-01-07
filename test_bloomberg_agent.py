"""Test Bloomberg quick search agent to verify domain filtering"""
import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"
ADMIN_USER = "admin"
ADMIN_PASSWORD = "admin123"

def login_and_get_session():
    """Login and get session"""
    session = requests.Session()
    
    # Get login page first
    session.get(f"{BASE_URL}/login")
    
    # Login
    login_resp = session.post(
        f"{BASE_URL}/login",
        data={'user_id': ADMIN_USER, 'password': ADMIN_PASSWORD},
        allow_redirects=False
    )
    
    if login_resp.status_code == 302:
        print("✓ Logged in successfully")
        return session
    else:
        print(f"✗ Login failed: {login_resp.status_code}")
        return None

def find_bloomberg_agent(session):
    """Find the Bloomberg quick search agent"""
    resp = session.get(f"{BASE_URL}/api/admin/agents")
    if resp.status_code != 200:
        print(f"✗ Failed to get agents: {resp.status_code}")
        return None
    
    data = resp.json()
    agents = data.get("agents", [])
    
    # Find Bloomberg agent (quick_search type) - look for "bbg" or "bloomberg" in name
    bloomberg_agent = None
    for agent in agents:
        if agent.get("agent_type") == "quick_search":
            name_lower = agent.get("name", "").lower()
            if "bloomberg" in name_lower or "bbg" in name_lower:
                bloomberg_agent = agent
                break
    
    if bloomberg_agent:
        print(f"✓ Found Bloomberg agent: {bloomberg_agent.get('name')} (ID: {bloomberg_agent.get('id')})")
        # Check config
        config = bloomberg_agent.get("quick_search_config", {})
        include_domains = config.get("include_domains")
        print(f"  Config - include_domains: {include_domains}")
        return bloomberg_agent.get("id")
    else:
        print("✗ Bloomberg quick search agent not found")
        # List all quick_search agents
        quick_search_agents = [a for a in agents if a.get("agent_type") == "quick_search"]
        if quick_search_agents:
            print(f"  Found {len(quick_search_agents)} quick_search agent(s):")
            for a in quick_search_agents:
                print(f"    - {a.get('name')} (ID: {a.get('id')})")
        return None

def test_query(session, agent_id):
    """Test a query with the Bloomberg agent"""
    query = "RBC stock price"
    print(f"\n🔍 Testing query: '{query}'")
    print(f"🤖 Agent ID: {agent_id}\n")
    
    # Start run
    resp = session.post(
        f"{BASE_URL}/api/agents/{agent_id}/runs",
        json={
            "query": query,
            "conversation_id": "test_bloomberg_" + str(int(time.time())),
            "show_thinking": False
        }
    )
    
    if resp.status_code != 200:
        print(f"✗ Failed to start run: {resp.status_code}")
        print(f"Response: {resp.text}")
        return False
    
    data = resp.json()
    run_id = data.get("run_id")
    if not run_id:
        print(f"✗ No run_id in response: {data}")
        return False
    
    print(f"✓ Run started: {run_id}")
    
    # Poll for completion
    start_time = time.time()
    timeout = 60
    
    while time.time() - start_time < timeout:
        time.sleep(2)
        status_resp = session.get(f"{BASE_URL}/api/runs/{run_id}")
        if status_resp.status_code != 200:
            print(f"✗ Failed to get status: {status_resp.status_code}")
            return False
        
        status_data = status_resp.json()
        current_status = status_data.get("status")
        
        if current_status in ["success", "error", "cancelled"]:
            print(f"\n✅ Run completed with status: {current_status}")
            
            if current_status == "success":
                result_summary = status_data.get("result_summary", "")
                print(f"\n📄 Response (first 1000 chars):\n{result_summary[:1000]}...")
                
                # Check if response contains only Bloomberg sources
                result_json = json.loads(result_summary) if result_summary.startswith("{") else {}
                sources = result_json.get("sources", [])
                
                print(f"\n📚 Sources found: {len(sources)}")
                non_bloomberg = []
                for i, source in enumerate(sources[:10], 1):
                    url = source.get("url", "")
                    title = source.get("title", "N/A")
                    is_bloomberg = "bloomberg.com" in url.lower()
                    print(f"  {i}. {title[:60]}...")
                    print(f"     URL: {url}")
                    print(f"     Bloomberg: {'✓' if is_bloomberg else '✗'}")
                    if not is_bloomberg:
                        non_bloomberg.append(url)
                
                if non_bloomberg:
                    print(f"\n⚠️  WARNING: Found {len(non_bloomberg)} non-Bloomberg source(s):")
                    for url in non_bloomberg:
                        print(f"     - {url}")
                    return False
                else:
                    print(f"\n✅ SUCCESS: All sources are from Bloomberg!")
                    return True
            else:
                print(f"✗ Run failed: {status_data.get('error_type', 'Unknown error')}")
                return False
    
    print(f"✗ Timeout after {timeout} seconds")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Bloomberg Quick Search Agent - Domain Filtering")
    print("=" * 60)
    
    session = login_and_get_session()
    if not session:
        exit(1)
    
    agent_id = find_bloomberg_agent(session)
    if not agent_id:
        exit(1)
    
    success = test_query(session, agent_id)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Domain filtering is working correctly")
    else:
        print("❌ TEST FAILED: Domain filtering is not working")
    print("=" * 60)

