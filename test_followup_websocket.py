"""
Test follow-up query detection via WebSocket API
"""
import socketio
import time
import json
import uuid

# Configuration
SERVER_URL = 'http://localhost:8000'
TOKEN = 'admin-token-placeholder'  # This will be replaced with actual token

# Create Socket.IO client with proper configuration
sio = socketio.Client(logger=False, engineio_logger=False)

# State tracking
session_id = f"test-{uuid.uuid4()}"
query_results = []
current_query = 0

@sio.on('connect')
def on_connect():
    print('✅ Connected to WebSocket')

@sio.on('disconnect')
def on_disconnect():
    print('🔌 Disconnected from WebSocket')

@sio.on('authenticated')
def on_authenticated(data):
    print(f'✅ Authenticated as: {data.get("user_id", "Unknown")}')

@sio.on('agent:node_start')
def on_node_start(data):
    print(f'  📍 Node: {data.get("node")}')

@sio.on('agent:node_complete')
def on_node_complete(data):
    pass  # Silent for cleaner output

@sio.on('agent:complete')
def on_complete(data):
    global current_query, query_results
    
    result = data.get('result', {})
    query_results.append(result)
    
    print(f'\n{"="*80}')
    print(f'✅ QUERY {current_query + 1} COMPLETE')
    print(f'{"="*80}')
    
    sql = result.get('plan_sql')
    if sql:
        print(f'SQL: {sql}')
        # Check for LIMIT clause
        if 'LIMIT' in sql.upper():
            limit_match = sql.upper().split('LIMIT')[-1].strip().split()[0]
            print(f'  → Has LIMIT: {limit_match}')
        else:
            print(f'  → No LIMIT clause (all rows)')
    
    execution = result.get('execution', {})
    if execution:
        print(f'Rows returned: {execution.get("row_count")}')
    
    clarification = result.get('clarification')
    if clarification:
        questions = clarification.get('questions', [])
        print(f'\n⚠️  Needs clarification ({len(questions)} questions)')
        for q in questions:
            print(f'  - {q}')

@sio.on('agent:waiting_for_clarification')
def on_waiting(data):
    print(f'⏸️  Agent paused for clarification')

@sio.on('agent:error')
def on_error(data):
    print(f'❌ Error: {data.get("error")}')

def send_query(query, wait_time=10):
    """Send a query and wait for completion"""
    global current_query
    
    print(f'\n{"="*80}')
    print(f'📤 SENDING QUERY {current_query + 1}')
    print(f'{"="*80}')
    print(f'Query: "{query}"')
    print(f'Session: {session_id}')
    
    sio.emit('agent:query', {
        'query': query,
        'session_id': session_id,
        'token': TOKEN
    })
    
    print(f'⏳ Waiting {wait_time}s for response...')
    time.sleep(wait_time)
    current_query += 1

def main():
    """Test automatic follow-up query detection"""
    
    try:
        # Step 1: Connect
        print(f'\n🚀 TESTING FOLLOW-UP QUERY DETECTION')
        print(f'Server: {SERVER_URL}')
        print(f'Session: {session_id}\n')
        
        print('🔌 Connecting...')
        sio.connect(SERVER_URL)
        time.sleep(1)
        
        # Step 2: Authenticate
        print('🔐 Authenticating...')
        sio.emit('authenticate', {'token': TOKEN})
        time.sleep(2)
        
        # Step 3: Send first query (with LIMIT 10)
        send_query("show me limits where utilization > 0.9", wait_time=12)
        
        # Step 4: Send follow-up query (should detect and combine)
        send_query("dont limit to 10 show all from above query", wait_time=12)
        
        # Analysis
        print(f'\n{"="*80}')
        print(f'📊 TEST ANALYSIS')
        print(f'{"="*80}')
        
        if len(query_results) >= 2:
            query1 = query_results[0]
            query2 = query_results[1]
            
            sql1 = query1.get('plan_sql', '')
            sql2 = query2.get('plan_sql', '')
            
            # Check Query 1
            print(f'\n✅ Query 1: {"PASS" if "LIMIT 10" in sql1 else "FAIL"}')
            print(f'   Expected: Has LIMIT 10')
            print(f'   Got: {"LIMIT 10" if "LIMIT 10" in sql1 else "No LIMIT or different LIMIT"}')
            
            # Check Query 2
            has_clarification = query2.get('clarification') is not None
            has_limit = 'LIMIT' in sql2.upper()
            
            print(f'\n✅ Query 2: {"PASS" if not has_clarification and not has_limit else "FAIL"}')
            print(f'   Expected: No clarification, no LIMIT clause')
            print(f'   Got: {"Clarification needed" if has_clarification else "SQL generated"}, {"Has LIMIT" if has_limit else "No LIMIT"}')
            
            if has_clarification:
                print(f'\n   ❌ FAIL: Follow-up query asked for clarification!')
                print(f'   This means automatic context combination did not work.')
            elif not has_limit:
                print(f'\n   ✅ SUCCESS: Follow-up query understood context!')
                print(f'   The agent correctly removed LIMIT from the query.')
            else:
                print(f'\n   ⚠️  PARTIAL: SQL generated but still has LIMIT clause.')
        
        else:
            print(f'\n❌ Incomplete: Only received {len(query_results)} results')
        
    except KeyboardInterrupt:
        print('\n⚠️  Test interrupted by user')
    except Exception as e:
        print(f'\n❌ Test error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if sio.connected:
            sio.disconnect()
        print('\n👋 Test complete')

if __name__ == '__main__':
    main()

