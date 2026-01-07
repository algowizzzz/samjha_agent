"""Test Deep Research backend API"""
import sys
import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from external.core.db.session import get_db_session
from external.agent.persistence import create_agent_db, get_agent_db

BASE_URL = "http://localhost:8000"

def create_test_agent():
    """Create a test deep research agent"""
    with get_db_session() as db:
        agent_id = "test_deep_research_agent"
        
        # Check if agent exists
        existing = get_agent_db(db, agent_id)
        if existing:
            print(f"✓ Test agent already exists: {agent_id}")
            return agent_id
        
        # Create new agent
        agent = create_agent_db(
            db,
            agent_id=agent_id,
            name="Test Deep Research Agent",
            agent_type="deep_research",
            description="Test agent for deep research integration",
            deep_research_config={
                "research_model": "anthropic:claude-3-5-haiku-20241022",
                "research_model_max_tokens": 8192,
                "summarization_model": "anthropic:claude-3-5-haiku-20241022",
                "summarization_model_max_tokens": 4096,
                "compression_model": "anthropic:claude-3-5-haiku-20241022",
                "compression_model_max_tokens": 4096,
                "final_report_model": "anthropic:claude-3-5-haiku-20241022",
                "final_report_model_max_tokens": 8192,
                "search_api": "tavily",
                "max_researcher_iterations": 2,  # Keep it short for testing
                "max_concurrent_research_units": 2,
                "allow_clarification": False
            }
        )
        db.commit()
        print(f"✓ Created test agent: {agent_id}")
        return agent_id

def login():
    """Login and get session token"""
    session = requests.Session()
    # Use web login endpoint which sets session cookie
    response = session.post(
        f"{BASE_URL}/login",
        data={"user_id": "admin", "password": "admin123"},
        allow_redirects=False
    )
    # Check if we got a redirect (success) or if cookie was set
    if response.status_code in [200, 302] or 'session' in session.cookies or len(session.cookies) > 0:
        print("✓ Logged in successfully")
        print(f"  Cookies: {list(session.cookies.keys())}")
        return session
    else:
        print(f"✗ Login failed: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        return None

def test_deep_research_query(session, agent_id):
    """Test deep research query"""
    query = "What are the latest developments in AI agent frameworks?"
    
    print(f"\n📝 Testing query: {query}")
    print(f"🤖 Agent ID: {agent_id}")
    
    # Start run
    response = session.post(
        f"{BASE_URL}/api/agents/{agent_id}/runs",
        json={
            "query": query,
            "show_thinking": False
        }
    )
    
    if response.status_code != 200:
        print(f"✗ Failed to start run: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return False
    
    try:
        data = response.json()
    except:
        print(f"✗ Response is not JSON: {response.text[:500]}")
        return False
    
    run_id = data.get("run_id")
    print(f"✓ Run started: {run_id}")
    
    # Poll for completion (with timeout)
    import time
    max_wait = 300  # 5 minutes
    start_time = time.time()
    last_status = None
    
    print("\n⏳ Waiting for completion...")
    while time.time() - start_time < max_wait:
        # Get run status
        status_response = session.get(f"{BASE_URL}/api/runs/{run_id}")
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data.get("status")
            
            if status != last_status:
                print(f"  Status: {status}")
                last_status = status
            
            if status in ["success", "error", "cancelled"]:
                print(f"\n✅ Run completed with status: {status}")
                
                # Get final result
                result_summary = status_data.get("result_summary", "")
                if result_summary:
                    try:
                        summary = json.loads(result_summary)
                        final_report = summary.get("final_report", "")
                        if final_report:
                            print(f"\n📄 Final Report (first 500 chars):")
                            print(final_report[:500] + "..." if len(final_report) > 500 else final_report)
                            print(f"\n📊 Report length: {len(final_report)} characters")
                            return True
                    except:
                        print(f"\n📄 Result Summary: {result_summary[:200]}...")
                
                return status == "success"
        
        time.sleep(2)
    
    print(f"\n⏱️  Timeout after {max_wait} seconds")
    return False

def main():
    print("=" * 60)
    print("Deep Research API Test")
    print("=" * 60)
    
    # Create test agent
    agent_id = create_test_agent()
    
    # Login
    session = login()
    if not session:
        return
    
    # Test query
    success = test_deep_research_query(session, agent_id)
    
    if success:
        print("\n✅ Test PASSED")
    else:
        print("\n❌ Test FAILED")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

