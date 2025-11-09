#!/usr/bin/env python3
"""
Test agent via UI API endpoint
Simulates what the UI does when calling the agent
"""
import requests
import json
import sys
from dotenv import load_dotenv

load_dotenv()

def test_agent_api():
    """Test agent via /api/tools/execute endpoint"""
    print("=" * 60)
    print("Testing Agent via UI API Endpoint")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Step 1: Login to get token
    print("\n1. Logging in...")
    login_data = {
        "user_id": "admin",
        "password": "admin123"  # Default from config/users.json
    }
    
    try:
        login_response = requests.post(
            f"{base_url}/api/auth/login",
            json=login_data,
            timeout=5
        )
        
        if login_response.status_code != 200:
            print(f"❌ Login failed: {login_response.status_code}")
            print(f"   Response: {login_response.text[:200]}")
            return False
        
        login_result = login_response.json()
        token = login_result.get('token')
        
        if not token:
            print("❌ No token received")
            return False
        
        print(f"✓ Login successful, token: {token[:20]}...")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        print("   Make sure server is running: python3 run_server.py")
        return False
    except Exception as e:
        print(f"❌ Login error: {e}")
        return False
    
    # Step 2: Test agent query
    print("\n2. Testing agent query (DATA QUERY)...")
    query = "top 10 limits by utilization"
    
    agent_request = {
        "tool": "parquet_agent",
        "arguments": {
            "query": query,
            "session_id": "test-session-ui-001",
            "user_id": "admin"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"   Query: {query}")
        print("   Sending request...")
        
        response = requests.post(
            f"{base_url}/api/tools/execute",
            json=agent_request,
            headers=headers,
            timeout=120  # Agent may take time
        )
        
        if response.status_code != 200:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
        
        result = response.json()
        
        if not result.get('success'):
            print(f"❌ Tool execution failed: {result.get('error', 'Unknown error')}")
            return False
        
        agent_result = result.get('result', {})
        
        print("\n✓ Agent query completed")
        print(f"   Session ID: {agent_result.get('session_id', 'N/A')}")
        print(f"   Control: {agent_result.get('control', 'N/A')}")
        print(f"   Plan Quality: {agent_result.get('plan_quality', 'N/A')}")
        
        # Check for final output
        final_output = agent_result.get('final_output', {})
        if final_output:
            response_text = final_output.get('response', '')
            print(f"\n   Response length: {len(response_text)} chars")
            print(f"   Response preview: {response_text[:200]}...")
            
            raw_table = final_output.get('raw_table', {})
            if raw_table:
                print(f"   Table rows: {raw_table.get('row_count', 0)}")
            
            prompt_monitor = final_output.get('prompt_monitor', {})
            if isinstance(prompt_monitor, dict) and 'procedural_reasoning' in prompt_monitor:
                print(f"   Prompt monitor: {len(prompt_monitor['procedural_reasoning'])} chars")
        
        # Check for clarification
        if agent_result.get('control') == 'wait_for_user':
            print(f"\n   ⚠ Clarification needed:")
            print(f"      {agent_result.get('clarify_prompt', 'N/A')}")
        
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (agent may be taking too long)")
        return False
    except Exception as e:
        print(f"❌ Request error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent_api()
    print("\n" + "=" * 60)
    if success:
        print("✓ API endpoint test PASSED")
        sys.exit(0)
    else:
        print("❌ API endpoint test FAILED")
        sys.exit(1)

