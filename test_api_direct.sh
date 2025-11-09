#!/bin/bash
# Test API directly
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin"}' | python3 -c "import sys, json; print(json.load(sys.stdin)['token'])")

echo "Testing API with query: sales revenue total"
echo "==========================================="

curl -s -X POST http://localhost:8000/api/tools/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "tool": "parquet_agent",
    "arguments": {
      "query": "sales revenue total",
      "session_id": "test-session",
      "user_id": "admin"
    }
  }' | python3 -c "
import sys, json
resp = json.load(sys.stdin)
print('Success:', resp.get('success'))
print('Has result:', 'result' in resp)
if 'result' in resp:
    result = resp['result']
    print('Has final_output:', 'final_output' in result)
    if 'final_output' in result:
        fo = result['final_output']
        print('Has response:', 'response' in fo)
        if 'response' in fo:
            print('Response preview:', fo['response'][:100])
"
