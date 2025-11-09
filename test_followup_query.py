"""
Test follow-up query detection with WebSocket
"""
import socketio
import time
import json

# Create a Socket.IO client
sio = socketio.Client()

session_id = None
query_count = 0

@sio.on('connect')
def on_connect():
    print('✅ Connected to WebSocket server')

@sio.on('authenticated')
def on_authenticated(data):
    print(f'✅ Authenticated as: {data.get("user_id")}')

@sio.on('agent:node_start')
def on_node_start(data):
    print(f'📍 Node started: {data.get("node")}')

@sio.on('agent:node_complete')
def on_node_complete(data):
    node = data.get('node')
    print(f'✅ Node completed: {node}')

@sio.on('agent:llm_chunk')
def on_llm_chunk(data):
    # Silent for cleaner output
    pass

@sio.on('agent:complete')
def on_complete(data):
    global query_count
    query_count += 1
    
    result = data.get('result', {})
    print(f'\n{"="*80}')
    print(f'🎯 QUERY {query_count} COMPLETE')
    print(f'{"="*80}')
    print(f'Session ID: {result.get("session_id")}')
    print(f'Control: {result.get("control")}')
    print(f'Plan Quality: {result.get("plan_quality")}')
    
    sql = result.get('plan_sql')
    if sql:
        print(f'\n📝 SQL Generated:')
        print(f'{sql}')
    
    execution = result.get('execution', {})
    if execution:
        print(f'\n📊 Execution:')
        print(f'  Row count: {execution.get("row_count")}')
    
    clarification = result.get('clarification')
    if clarification:
        questions = clarification.get('questions', [])
        print(f'\n❓ Clarification needed ({len(questions)} questions):')
        for i, q in enumerate(questions, 1):
            print(f'  {i}. {q}')

@sio.on('agent:waiting_for_clarification')
def on_waiting(data):
    print(f'\n⏸️  Agent waiting for clarification')

@sio.on('agent:error')
def on_error(data):
    print(f'\n❌ Error: {data.get("error")}')

def test_follow_up_queries():
    """Test automatic follow-up query detection"""
    global session_id
    
    try:
        # Connect to server
        print('\n🔌 Connecting to WebSocket server...')
        sio.connect('http://localhost:8000')
        time.sleep(1)
        
        # Authenticate
        print('🔐 Authenticating...')
        sio.emit('authenticate', {
            'token': 'admin-token-placeholder'  # Use your actual token
        })
        time.sleep(1)
        
        # Generate session ID
        import uuid
        session_id = f"test-{uuid.uuid4()}"
        print(f'📋 Session ID: {session_id}')
        
        # Query 1: Initial query with default LIMIT 10
        print('\n' + '='*80)
        print('TEST 1: Initial query with default LIMIT 10')
        print('='*80)
        query1 = "show me limits where utilization > 0.9"
        print(f'Query: {query1}')
        
        sio.emit('agent:query', {
            'query': query1,
            'session_id': session_id,
            'token': 'admin-token-placeholder'
        })
        
        # Wait for completion
        time.sleep(8)
        
        # Query 2: Follow-up query to remove limit
        print('\n' + '='*80)
        print('TEST 2: Follow-up query (should auto-combine with Query 1)')
        print('='*80)
        query2 = "dont limit to 10 show all from above query"
        print(f'Query: {query2}')
        print('Expected: Should detect follow-up and combine with previous query')
        
        sio.emit('agent:query', {
            'query': query2,
            'session_id': session_id,
            'token': 'admin-token-placeholder'
        })
        
        # Wait for completion
        time.sleep(8)
        
        print('\n' + '='*80)
        print('✅ TEST COMPLETE')
        print('='*80)
        print('\nExpected behavior:')
        print('  Query 1: Should return 8 rows with LIMIT 10 in SQL')
        print('  Query 2: Should return all rows WITHOUT LIMIT in SQL')
        print('  Query 2: Should NOT ask for clarification about "above query"')
        
    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect
        if sio.connected:
            sio.disconnect()
            print('\n🔌 Disconnected')

if __name__ == '__main__':
    test_follow_up_queries()

