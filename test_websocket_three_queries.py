#!/usr/bin/env python3
"""
Test WebSocket with three types of queries:
1. Knowledge question
2. Ambiguous query  
3. Clear query
"""
import requests
import time

def get_token():
    """Get auth token"""
    response = requests.post("http://localhost:8000/api/auth/login", 
                            json={"user_id": "admin", "password": "admin123"})
    if response.status_code == 200:
        return response.json().get('token')
    return None

def test_query(query, description):
    """Test a single query via REST API"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Query: {query}")
    print('='*60)
    
    token = get_token()
    if not token:
        print("❌ Failed to get token")
        return
    
    response = requests.post(
        "http://localhost:8000/api/tools/execute",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "tool": "parquet_agent",
            "arguments": {
                "query": query,
                "session_id": f"test-{int(time.time())}",
                "user_id": "admin"
            }
        },
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json().get('result', {})
        control = result.get('control')
        plan_quality = result.get('plan_quality')
        
        print(f"\n✓ Status: {control}")
        print(f"  Plan Quality: {plan_quality}")
        
        if control == "wait_for_user":
            clarification = result.get('clarification', {})
            questions = clarification.get('questions', [])
            print(f"  ✋ CLARIFICATION NEEDED ({len(questions)} questions):")
            print(f"  DEBUG - Full result keys: {list(result.keys())}")
            print(f"  DEBUG - clarification_questions from result: {result.get('clarification_questions', 'NOT FOUND')}")
            print(f"  DEBUG - Full clarification: {clarification}")
            for i, q in enumerate(questions, 1):
                print(f"     {i}. {q}")
        elif control == "end":
            final = result.get('final_output', {})
            response_text = final.get('response', '')
            print(f"  ✓ COMPLETED")
            print(f"  Response length: {len(response_text)} chars")
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"   {response.text[:200]}")

if __name__ == "__main__":
    # Test 1: Knowledge question
    test_query("what is vega", "Knowledge Question")
    
    # Test 2: Ambiguous query
    test_query("give me large breaches", "Ambiguous Query")
    
    # Test 3: Clear query
    test_query("SELECT * FROM limits_data WHERE utilization > 0.9 ORDER BY utilization DESC LIMIT 5", 
               "Clear SQL Query")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

