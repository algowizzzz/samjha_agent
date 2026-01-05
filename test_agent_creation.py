#!/usr/bin/env python3
"""
Test script to verify agent creation API works correctly.
This simulates what the frontend sends.
"""
import requests
import json
from pathlib import Path

def test_agent_creation():
    """Test creating a structured agent via API"""
    print("=" * 60)
    print("TESTING AGENT CREATION API")
    print("=" * 60)
    print()
    
    # Create a test domain file
    test_domain = Path('test_domain.md')
    test_domain.write_text('# Test Domain\n\nThis is a test domain configuration.')
    
    url = 'http://localhost:5000/api/admin/agents'
    
    # Simulate what the frontend sends
    files = {
        'domain_file': ('test_domain.md', open('test_domain.md', 'rb'), 'text/markdown')
    }
    
    data = {
        'name': 'Test Structured Agent API',
        'description': 'Test agent created via API',
        'agent_type': 'structured',  # This is the critical field
        'model': 'claude-sonnet-4-20250514',
        'data_folder_mode': 'create',
        'data_folder_name': 'test_agent_data_api'
    }
    
    print("Request details:")
    print(f"  URL: {url}")
    print(f"  Method: POST")
    print(f"  agent_type: {data['agent_type']}")
    print(f"  name: {data['name']}")
    print()
    
    try:
        response = requests.post(url, files=files, data=data, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200 or response.status_code == 201:
            result = response.json()
            print("✓ SUCCESS!")
            print(f"  Agent ID: {result.get('id', 'N/A')}")
            print(f"  Agent Name: {result.get('name', 'N/A')}")
            print(f"  Agent Type: {result.get('agent_type', 'N/A')}")
        elif response.status_code == 403:
            print("⚠️  Authentication required (expected if not logged in)")
            print("   This means the endpoint is working, but needs login")
        elif response.status_code == 400:
            error = response.json() if response.content else {}
            print("✗ VALIDATION ERROR:")
            print(f"  {error.get('error', response.text)}")
            if 'agent_type' in str(error.get('error', '')).lower():
                print()
                print("  This indicates the agent_type validation failed.")
                print("  Check if agent_type is being sent correctly.")
        else:
            print(f"✗ ERROR: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server.")
        print("  Make sure the Flask server is running on port 5000")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        # Cleanup
        test_domain.unlink()
        if 'files' in locals() and files.get('domain_file'):
            files['domain_file'][1].close()
    
    print()
    print("=" * 60)

if __name__ == '__main__':
    test_agent_creation()

