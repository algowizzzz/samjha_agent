# Streaming Response, Kill Button, and Loading Visual - Implementation Plan

## Overview
Add real-time streaming of agent responses, ability to cancel running queries, and visual feedback during processing.

---

## Step 1: Backend - Add Session Management & Cancellation

### 1.1 Create Session Manager
**File:** `agent/session_manager.py` (new file)

```python
import threading
import uuid
from typing import Dict, Optional
from datetime import datetime

class AgentSessionManager:
    """Manages active agent sessions and cancellation"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}  # session_id -> {thread, cancelled, start_time}
        self._lock = threading.Lock()
    
    def register_session(self, session_id: str, thread: threading.Thread) -> None:
        """Register an active session"""
        with self._lock:
            self.active_sessions[session_id] = {
                'thread': thread,
                'cancelled': False,
                'start_time': datetime.utcnow()
            }
    
    def cancel_session(self, session_id: str) -> bool:
        """Mark session as cancelled"""
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['cancelled'] = True
                return True
            return False
    
    def is_cancelled(self, session_id: str) -> bool:
        """Check if session is cancelled"""
        with self._lock:
            return self.active_sessions.get(session_id, {}).get('cancelled', False)
    
    def unregister_session(self, session_id: str) -> None:
        """Remove session from active list"""
        with self._lock:
            self.active_sessions.pop(session_id, None)
```

### 1.2 Modify ParquetQueryAgent for Streaming & Cancellation
**File:** `agent/parquet_agent.py`

**Changes:**
1. Add `session_manager` parameter to `__init__`
2. Add `stream_callback` parameter to `run_query()` method
3. Check `is_cancelled()` at each node
4. Call `stream_callback()` with progress updates
5. Yield chunks from `end_node` LLM calls

**Key modifications:**
```python
def run_query(self, user_input: str, user_id: Optional[str] = None, 
              session_id: Optional[str] = None,
              stream_callback=None, session_manager=None):
    """Run query with optional streaming"""
    
    # Check cancellation at start
    if session_manager and session_manager.is_cancelled(session_id):
        return {"error": "Query cancelled", "session_id": session_id}
    
    # Stream progress updates
    if stream_callback:
        stream_callback({"type": "status", "message": "Starting query..."})
    
    # In each node, check cancellation and stream updates
    # ...
```

---

## Step 2: Backend - Add Streaming Endpoint

### 2.1 Add WebSocket Route for Streaming
**File:** `routes/socketio_handlers.py`

**Add new handler:**
```python
@self.socketio.on('agent_query_stream')
def handle_agent_query_stream(data):
    """Handle streaming agent query via WebSocket"""
    session_id = data.get('session_id')
    query = data.get('query')
    user_id = data.get('user_id')
    
    # Create session manager instance
    session_manager = AgentSessionManager()
    
    def stream_callback(chunk):
        """Stream chunks back to client"""
        emit('agent_chunk', {
            'session_id': session_id,
            'chunk': chunk
        })
    
    # Run agent in background thread
    def run_agent():
        agent = ParquetQueryAgent()
        try:
            result = agent.run_query(
                query, 
                user_id=user_id,
                session_id=session_id,
                stream_callback=stream_callback,
                session_manager=session_manager
            )
            emit('agent_complete', {
                'session_id': session_id,
                'result': result
            })
        except Exception as e:
            emit('agent_error', {
                'session_id': session_id,
                'error': str(e)
            })
        finally:
            session_manager.unregister_session(session_id)
    
    thread = threading.Thread(target=run_agent)
    session_manager.register_session(session_id, thread)
    thread.start()
```

### 2.2 Add Kill Handler
**File:** `routes/socketio_handlers.py`

```python
@self.socketio.on('agent_kill')
def handle_agent_kill(data):
    """Cancel a running agent query"""
    session_id = data.get('session_id')
    if session_manager.cancel_session(session_id):
        emit('agent_killed', {'session_id': session_id})
    else:
        emit('agent_kill_error', {'error': 'Session not found'})
```

---

## Step 3: Backend - Modify Graph Nodes for Streaming

### 3.1 Update end_node for LLM Streaming
**File:** `agent/graph_nodes.py`

**Modify LLM calls to support streaming:**
```python
# In end_node, modify LLM calls:
if llm_client.is_available():
    # Stream LLM response chunk by chunk
    if stream_callback:
        for chunk in llm_client.invoke_with_prompt_streaming(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3
        ):
            if session_manager and session_manager.is_cancelled(state.get('session_id')):
                break
            stream_callback({
                "type": "response_chunk",
                "chunk": chunk,
                "field": "response"  # or "prompt_monitor"
            })
            llm_response += chunk
    else:
        # Non-streaming fallback
        llm_response = llm_client.invoke_with_prompt(...)
```

### 3.2 Add Streaming Support to LLM Client
**File:** `agent/llm_client.py`

**Add streaming method:**
```python
def invoke_with_prompt_streaming(self, system_prompt: str, user_prompt: str, 
                                  temperature: float = 0.7):
    """Stream LLM response chunk by chunk"""
    if not self.is_available():
        raise RuntimeError("LLM not available")
    
    # For Anthropic, use stream parameter
    with self.client.messages.stream(
        model=self.model_name,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature
    ) as stream:
        for text in stream.text_stream:
            yield text
```

---

## Step 4: Frontend - Add Loading Visual

### 4.1 Add Loading Spinner Component
**File:** `web/templates/agent_chat.html`

**Add CSS:**
```css
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(0,0,0,.1);
    border-radius: 50%;
    border-top-color: #007bff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.agent-thinking {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background-color: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 12px;
}
```

**Add loading HTML function:**
```javascript
function showLoading(sessionId) {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading_' + sessionId;
    loadingDiv.className = 'agent-thinking';
    loadingDiv.innerHTML = `
        <div class="loading-spinner"></div>
        <span>Agent is thinking...</span>
        <button class="btn btn-sm btn-danger ms-auto" onclick="killAgent('${sessionId}')">
            <i class="bi bi-x-circle"></i> Stop
        </button>
    `;
    historyEl.appendChild(loadingDiv);
    historyEl.scrollTop = historyEl.scrollHeight;
}

function hideLoading(sessionId) {
    const loadingEl = document.getElementById('loading_' + sessionId);
    if (loadingEl) {
        loadingEl.remove();
    }
}
```

---

## Step 5: Frontend - Add WebSocket Connection

### 5.1 Connect to WebSocket
**File:** `web/templates/agent_chat.html`

**Add WebSocket connection:**
```javascript
// Connect to SocketIO
const socket = io();

socket.on('connect', function() {
    console.log('WebSocket connected');
    // Authenticate
    socket.emit('authenticate', {
        token: templateToken || sessionStorage.getItem('mcp_token')
    });
});

socket.on('authenticated', function(data) {
    console.log('WebSocket authenticated');
});

// Listen for streaming chunks
socket.on('agent_chunk', function(data) {
    handleStreamChunk(data);
});

socket.on('agent_complete', function(data) {
    handleStreamComplete(data);
});

socket.on('agent_error', function(data) {
    handleStreamError(data);
});

socket.on('agent_killed', function(data) {
    handleAgentKilled(data);
});
```

---

## Step 6: Frontend - Handle Streaming Updates

### 6.1 Add Streaming Handler Functions
**File:** `web/templates/agent_chat.html`

```javascript
let currentStreamingSession = null;
let streamingMessageDiv = null;
let streamingResponse = '';
let streamingPromptMonitor = '';

function handleStreamChunk(data) {
    const { session_id, chunk } = data;
    
    if (chunk.type === 'status') {
        // Update loading message
        updateLoadingMessage(session_id, chunk.message);
    } else if (chunk.type === 'response_chunk') {
        // Append to streaming response
        if (chunk.field === 'response') {
            streamingResponse += chunk.chunk;
            updateStreamingResponse(session_id, streamingResponse);
        } else if (chunk.field === 'prompt_monitor') {
            streamingPromptMonitor += chunk.chunk;
            // Can update prompt monitor in real-time too
        }
    }
}

function updateStreamingResponse(sessionId, text) {
    if (!streamingMessageDiv) {
        // Create message div if it doesn't exist
        streamingMessageDiv = document.createElement('div');
        streamingMessageDiv.className = 'chat-message agent';
        streamingMessageDiv.innerHTML = '<span class="badge bg-success me-2">Agent</span><div class="markdown-body"></div>';
        historyEl.appendChild(streamingMessageDiv);
    }
    
    // Render markdown as it streams
    const markdownDiv = streamingMessageDiv.querySelector('.markdown-body');
    if (typeof marked !== 'undefined') {
        markdownDiv.innerHTML = marked.parse(text);
    } else {
        markdownDiv.textContent = text;
    }
    
    historyEl.scrollTop = historyEl.scrollHeight;
}

function handleStreamComplete(data) {
    const { session_id, result } = data;
    
    hideLoading(session_id);
    
    // Finalize the message with full output
    if (streamingMessageDiv) {
        const finalOut = result.final_output || {};
        
        // Update with final response
        if (finalOut.response) {
            updateStreamingResponse(session_id, finalOut.response);
        }
        
        // Add raw table and prompt monitor
        appendFinalOutput(streamingMessageDiv, finalOut);
        
        streamingMessageDiv = null;
        streamingResponse = '';
        streamingPromptMonitor = '';
    }
    
    currentStreamingSession = null;
}

function handleStreamError(data) {
    const { session_id, error } = data;
    hideLoading(session_id);
    appendMessage('agent', 'Error: ' + error);
    currentStreamingSession = null;
}

function handleAgentKilled(data) {
    const { session_id } = data;
    hideLoading(session_id);
    if (streamingMessageDiv) {
        streamingMessageDiv.innerHTML += '<div class="alert alert-warning mt-2">Query was cancelled by user.</div>';
    }
    currentStreamingSession = null;
}
```

---

## Step 7: Frontend - Add Kill Button Functionality

### 7.1 Implement Kill Function
**File:** `web/templates/agent_chat.html`

```javascript
function killAgent(sessionId) {
    if (!sessionId || !currentStreamingSession) {
        return;
    }
    
    // Emit kill event
    socket.emit('agent_kill', {
        session_id: sessionId
    });
    
    // Update UI immediately
    const killBtn = document.querySelector(`#loading_${sessionId} button`);
    if (killBtn) {
        killBtn.disabled = true;
        killBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Stopping...';
    }
}

// Update sendQuery to use WebSocket
function sendQuery() {
    const q = inputEl.value.trim();
    if (!q) return;
    
    appendMessage('user', q);
    inputEl.value = '';
    $('#sendBtn').prop('disabled', true);
    
    // Generate session ID
    const sessionId = 'sess-' + Math.random().toString(36).slice(2, 10);
    currentStreamingSession = sessionId;
    
    // Show loading
    showLoading(sessionId);
    
    // Reset streaming state
    streamingMessageDiv = null;
    streamingResponse = '';
    streamingPromptMonitor = '';
    
    // Emit query via WebSocket
    socket.emit('agent_query_stream', {
        session_id: sessionId,
        query: q,
        user_id: '{{ user.user_id | default("anonymous") }}',
        user_clarification: awaitingClarification ? q : null
    });
    
    if (awaitingClarification) {
        awaitingClarification = false;
        inputEl.placeholder = 'Ask a question or provide a follow-up...';
    }
}
```

---

## Step 8: Update sendQuery to Use Streaming

### 8.1 Modify Existing sendQuery Function
**File:** `web/templates/agent_chat.html`

**Replace the existing `sendQuery()` function** with the WebSocket version above, or add a toggle to switch between REST and WebSocket modes.

---

## Step 9: Testing Checklist

1. **Test Streaming:**
   - Send a query
   - Verify response appears chunk by chunk
   - Verify markdown renders as it streams

2. **Test Kill Button:**
   - Start a long-running query
   - Click "Stop" button
   - Verify query stops and shows cancellation message

3. **Test Loading Visual:**
   - Verify spinner appears when query starts
   - Verify spinner disappears when query completes
   - Verify status messages update during processing

4. **Test Error Handling:**
   - Test with invalid query
   - Test network disconnection
   - Test server error

---

## Step 10: Optional Enhancements

1. **Progress Bar:** Show percentage complete based on node execution
2. **Node Status:** Show which node is currently executing (planner, executor, etc.)
3. **Retry Button:** Allow retrying failed queries
4. **Streaming Table:** Stream table rows as they're generated
5. **Typing Indicator:** Show "Agent is typing..." animation

---

## Implementation Order

1. ✅ Step 1: Backend session management
2. ✅ Step 2: Backend streaming endpoint
3. ✅ Step 3: Backend graph node modifications
4. ✅ Step 4: Frontend loading visual
5. ✅ Step 5: Frontend WebSocket connection
6. ✅ Step 6: Frontend streaming handlers
7. ✅ Step 7: Frontend kill button
8. ✅ Step 8: Update sendQuery
9. ✅ Step 9: Testing
10. ✅ Step 10: Optional enhancements

---

## Files to Create/Modify

**New Files:**
- `agent/session_manager.py`

**Modified Files:**
- `agent/parquet_agent.py`
- `agent/graph_nodes.py`
- `agent/llm_client.py`
- `routes/socketio_handlers.py`
- `web/templates/agent_chat.html`

---

## Dependencies

- Flask-SocketIO (already installed)
- Threading support (Python standard library)
- WebSocket client (Socket.IO JavaScript library - already in use)

