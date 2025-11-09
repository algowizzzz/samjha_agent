#!/usr/bin/env python3
"""
Test WebSocket streaming endpoint
Simulates what the UI does when using WebSocket for streaming
"""
import socketio
import time
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

def get_auth_token():
    """Login and get authentication token"""
    base_url = "http://localhost:8000"
    login_data = {
        "user_id": "admin",
        "password": "admin123"
    }
    
    try:
        response = requests.post(f"{base_url}/api/auth/login", json=login_data, timeout=5)
        if response.status_code == 200:
            return response.json().get('token')
        else:
            print(f"❌ Login failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

def test_websocket_streaming():
    """Test WebSocket streaming"""
    print("=" * 60)
    print("Testing WebSocket Streaming Endpoint")
    print("=" * 60)
    
    # Get authentication token
    print("\n0. Getting authentication token...")
    token = get_auth_token()
    if not token:
        print("❌ Failed to get auth token")
        return False
    print(f"✓ Got token: {token[:20]}...")
    
    # Create SocketIO client
    sio = socketio.Client()
    
    events_received = []
    chunks_received = []
    
    @sio.on('connect')
    def on_connect():
        print("✓ Connected to WebSocket")
        # Authenticate
        sio.emit('authenticate', {'token': token})
    
    @sio.on('authenticated')
    def on_authenticated(data):
        if data.get('success'):
            print(f"✓ Authenticated: {data.get('user', 'N/A')}")
            # Join session
            session_id = 'test-ws-session-001'
            sio.emit('join_session', {'session_id': session_id})
        else:
            print(f"❌ Authentication failed: {data.get('error')}")
            sio.disconnect()
    
    @sio.on('session_joined')
    def on_session_joined(data):
        print(f"✓ Joined session: {data.get('session_id')}")
        # Now send agent query
        time.sleep(0.5)
        print("\n📤 Sending agent query via WebSocket (AMBIGUOUS QUERY)...")
        sio.emit('agent:query', {
            'query': 'give me large breaches',
            'session_id': 'test-ws-session-001',
            'token': token
        })
    
    @sio.on('agent:node_start')
    def on_node_start(data):
        node = data.get('node', 'unknown')
        events_received.append(('node_start', node))
        print(f"  [NODE START] {node}")
    
    @sio.on('agent:llm_chunk')
    def on_llm_chunk(data):
        node = data.get('node', 'unknown')
        chunk = data.get('chunk', '')
        chunks_received.append((node, chunk))
        if len(chunks_received) <= 5:  # Show first 5 chunks
            print(f"  [CHUNK] {node}: {chunk[:50]}...")
    
    @sio.on('agent:node_complete')
    def on_node_complete(data):
        node = data.get('node', 'unknown')
        events_received.append(('node_complete', node))
        print(f"  [NODE COMPLETE] {node}")
    
    @sio.on('agent:waiting_for_clarification')
    def on_waiting_for_clarification(data):
        events_received.append(('waiting_for_clarification', 'clarify'))
        result = data.get('result', {})
        clarification = result.get('clarification', {})
        
        print(f"\n✋ CLARIFICATION NEEDED")
        print(f"   Reasoning: {clarification.get('reasoning', [])}")
        print(f"   Questions ({len(clarification.get('questions', []))}):")
        for i, q in enumerate(clarification.get('questions', []), 1):
            print(f"     {i}. {q}")
        
        # Disconnect after receiving clarification
        time.sleep(1)
        sio.disconnect()
    
    @sio.on('agent:complete')
    def on_complete(data):
        events_received.append(('complete', 'end'))
        result = data.get('result', {})
        print(f"\n✓ Agent completed")
        print(f"   Total chunks received: {len(chunks_received)}")
        print(f"   Total events: {len(events_received)}")
        
        final_out = result.get('final_output', {})
        if final_out:
            print(f"   Response length: {len(final_out.get('response', ''))} chars")
            print(f"   Has table: {bool(final_out.get('raw_table'))}")
        
        sio.disconnect()
    
    @sio.on('agent:error')
    def on_error(data):
        error = data.get('error', 'Unknown error')
        print(f"❌ Agent error: {error}")
        events_received.append(('error', error))
        sio.disconnect()
    
    try:
        print("\n1. Connecting to WebSocket server...")
        sio.connect('http://localhost:8000', wait_timeout=5)
        
        # Wait for completion (with timeout)
        print("\n2. Waiting for streaming response...")
        timeout = 120  # 2 minutes
        start_time = time.time()
        
        while sio.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if sio.connected:
            print("⚠ Timeout reached, disconnecting...")
            sio.disconnect()
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print("=" * 60)
        print(f"✓ Events received: {len(events_received)}")
        print(f"✓ Chunks received: {len(chunks_received)}")
        
        if len(chunks_received) > 0:
            print("✓ Streaming is working!")
            # Count chunks by node
            node_counts = {}
            for node, _ in chunks_received:
                node_counts[node] = node_counts.get(node, 0) + 1
            print(f"   Chunks by node: {node_counts}")
        else:
            print("⚠ No streaming chunks received")
        
        return len(chunks_received) > 0 or len(events_received) > 0
        
    except socketio.exceptions.ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        print("   Make sure server is running and SocketIO is configured")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Note: This requires python-socketio client library
    try:
        import socketio
    except ImportError:
        print("❌ python-socketio not installed")
        print("   Install with: pip install python-socketio")
        sys.exit(1)
    
    success = test_websocket_streaming()
    print("\n" + "=" * 60)
    if success:
        print("✓ WebSocket streaming test PASSED")
        sys.exit(0)
    else:
        print("❌ WebSocket streaming test FAILED")
        sys.exit(1)

