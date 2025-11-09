# Streaming Support for Parquet Query Agent

## Overview

Add streaming support to the agent to provide real-time feedback during LLM calls, improving perceived performance and user experience. Stream LLM responses as they're generated instead of waiting for complete responses.

## Current Architecture

### LLM Calls (Synchronous)
- **Planner Node**: `tools/impl/nl_to_sql_planner.py` → `_llm_plan()` → `LLMClient.invoke()`
- **Evaluator Node**: `tools/impl/query_result_evaluator.py` → `_llm_evaluate()` → `LLMClient.invoke()`
- **End Node**: `agent/graph_nodes.py` → `end_node()` → 2 LLM calls (response + prompt monitor)

### Communication Flow
- **UI**: `agent_chat.html` → HTTP POST `/api/tools/execute` → `agent/base_agent.py` → `ParquetQueryAgent.run_query()`
- **WebSocket**: SocketIO already configured but not used for agent streaming

## Streaming Strategy

### Option 1: WebSocket Streaming (Recommended)
- Use existing SocketIO infrastructure
- Stream chunks in real-time as LLM generates tokens
- Better for bidirectional communication
- Supports multiple concurrent sessions

### Option 2: Server-Sent Events (SSE)
- Simpler implementation
- One-way streaming (server → client)
- Requires new endpoint

**Decision: Use WebSocket (Option 1)** - Already have SocketIO, better for future features

## Implementation Plan

### Phase 1: Add Streaming Support to LLMClient

**File:** `agent/llm_client.py`

**Changes:**

1. **Add streaming method to LLMClient:**
   ```python
   def stream(
       self,
       messages: List[Dict[str, str]],
       system: Optional[str] = None,
       temperature: Optional[float] = None,
       max_tokens: Optional[int] = None,
       response_format: Optional[str] = None,
       callback: Optional[Callable[[str], None]] = None
   ) -> Generator[str, None, None]:
       """
       Stream LLM response token by token.
       
       Args:
           messages: List of message dicts
           system: Optional system prompt
           temperature: Override default
           max_tokens: Override default
           response_format: "json" for JSON output
           callback: Optional callback function for each chunk
       
       Yields:
           Text chunks as they arrive
       """
   ```

2. **Implement `_stream_anthropic` method:**
   - Use `client.messages.stream()` from Anthropic SDK
   - Yield text deltas from `stream.text_delta`
   - Call callback if provided
   - Accumulate full response for return

3. **Add `stream_with_prompt` convenience method:**
   ```python
   def stream_with_prompt(
       self,
       system_prompt: str,
       user_prompt: str,
       temperature: Optional[float] = None,
       response_format: Optional[str] = None,
       callback: Optional[Callable[[str], None]] = None
   ) -> Generator[str, None, None]:
   ```

**Testing:**
- Unit test streaming with mock callback
- Verify full response matches non-streaming version

---

### Phase 2: Create Streaming Agent Wrapper

**File:** `agent/streaming_agent.py` (new)

**Purpose:** Wrapper around `ParquetQueryAgent` that supports streaming

**Key Components:**

1. **StreamingAgent class:**
   ```python
   class StreamingAgent:
       def __init__(self, agent: ParquetQueryAgent, socketio, session_id: str):
           self.agent = agent
           self.socketio = socketio
           self.session_id = session_id
       
       def run_query_streaming(
           self,
           query: str,
           user_id: Optional[str] = None,
           emit_callback: Optional[Callable] = None
       ) -> Dict[str, Any]:
           """
           Run agent query with streaming LLM responses.
           
           Emits events:
           - 'agent:node_start': Node beginning execution
           - 'agent:llm_chunk': LLM text chunk (for streaming nodes)
           - 'agent:node_complete': Node finished
           - 'agent:state_update': State changed
           - 'agent:complete': Final result
           - 'agent:error': Error occurred
           """
   ```

2. **Streaming callbacks:**
   - Wrap LLM calls in planner/evaluator/end nodes
   - Emit chunks via SocketIO to specific session
   - Track which node is streaming (planner, evaluator, end_response, end_prompt_monitor)

3. **State management:**
   - Emit state updates after each node
   - Allow UI to show progress (e.g., "Planning...", "Evaluating...", "Generating response...")

**Event Schema:**
```json
{
  "event": "agent:llm_chunk",
  "data": {
    "node": "planner|evaluator|end_response|end_prompt_monitor",
    "chunk": "text chunk",
    "accumulated": "full text so far",
    "session_id": "session-id"
  }
}
```

---

### Phase 3: Modify Graph Nodes for Streaming

**Files to modify:**
- `tools/impl/nl_to_sql_planner.py`
- `tools/impl/query_result_evaluator.py`
- `agent/graph_nodes.py`

**Approach:** Add optional `stream_callback` parameter to LLM calls

**Changes:**

1. **NLToSQLPlannerTool._llm_plan():**
   ```python
   def _llm_plan(
       self,
       nl_query: str,
       schema_summary: str,
       business_context: str,
       procedural_knowledge: str,
       stream_callback: Optional[Callable[[str], None]] = None
   ) -> Dict[str, Any]:
       # If stream_callback provided, use streaming
       if stream_callback:
           llm_client = get_llm_client()
           full_response = ""
           for chunk in llm_client.stream(...):
               full_response += chunk
               stream_callback(chunk)
           # Parse full_response
       else:
           # Existing non-streaming code
   ```

2. **QueryResultEvaluatorTool._llm_evaluate():**
   - Same pattern: add `stream_callback` parameter
   - Use streaming if callback provided

3. **end_node() in graph_nodes.py:**
   - Add `stream_callback` parameter
   - Support two separate streams: one for response, one for prompt monitor
   - Emit with node identifiers: `end_response` and `end_prompt_monitor`

**Backward Compatibility:**
- All streaming parameters are optional
- Non-streaming calls work unchanged
- Default behavior: no streaming

---

### Phase 4: Add WebSocket Handler for Streaming Agent

**File:** `routes/socketio_handlers.py`

**Changes:**

1. **Add new SocketIO event handler:**
   ```python
   @self.socketio.on('agent:query')
   def handle_agent_query(data):
       """Handle streaming agent query"""
       # Validate session
       token = data.get('token')
       session_data = self.auth_manager.validate_session(token)
       if not session_data:
           emit('agent:error', {'error': 'Unauthorized'})
           return
       
       # Get query and session_id
       query = data.get('query')
       session_id = data.get('session_id')
       user_id = session_data.get('user_id')
       
       # Get agent tool
       agent_tool = self.tools_registry.get_tool('parquet_query_agent')
       if not agent_tool:
           emit('agent:error', {'error': 'Agent tool not found'})
           return
       
       # Create streaming wrapper
       from agent.streaming_agent import StreamingAgent
       streaming_agent = StreamingAgent(
           agent=agent_tool.agent,
           socketio=self.socketio,
           session_id=session_id
       )
       
       # Run query with streaming (in background thread)
       import threading
       def run_agent():
           try:
               result = streaming_agent.run_query_streaming(
                   query=query,
                   user_id=user_id
               )
               emit('agent:complete', {'result': result})
           except Exception as e:
               emit('agent:error', {'error': str(e)})
       
       thread = threading.Thread(target=run_agent, daemon=True)
       thread.start()
   ```

2. **Add helper method for emitting chunks:**
   ```python
   def _emit_agent_chunk(self, session_id: str, node: str, chunk: str, accumulated: str):
       """Emit LLM chunk to specific session"""
       self.socketio.emit('agent:llm_chunk', {
           'node': node,
           'chunk': chunk,
           'accumulated': accumulated,
           'session_id': session_id
       }, room=session_id)  # Emit only to this session
   ```

**Room-based emission:**
- Use SocketIO rooms to isolate sessions
- Join session_id room when client connects
- Emit chunks only to that room

---

### Phase 5: Update UI for Streaming

**File:** `web/templates/agent_chat.html`

**Changes:**

1. **Connect to WebSocket on page load:**
   ```javascript
   const socket = io();
   let currentStreamingNode = null;
   let accumulatedText = {
       planner: '',
       evaluator: '',
       end_response: '',
       end_prompt_monitor: ''
   };
   
   socket.on('connect', () => {
       console.log('Connected to WebSocket');
       // Authenticate
       socket.emit('authenticate', { token: getToken() });
   });
   
   socket.on('authenticated', (data) => {
       if (data.success) {
           // Join session room
           socket.emit('join_session', { session_id: sessionId });
       }
   });
   ```

2. **Handle streaming events:**
   ```javascript
   socket.on('agent:node_start', (data) => {
       // Show node status (e.g., "Planning query...")
       updateNodeStatus(data.node);
   });
   
   socket.on('agent:llm_chunk', (data) => {
       // Append chunk to appropriate UI element
       if (data.node === 'end_response') {
           appendToResponse(data.chunk);
       } else if (data.node === 'end_prompt_monitor') {
           appendToPromptMonitor(data.chunk);
       }
       // Update accumulated text
       accumulatedText[data.node] = data.accumulated;
   });
   
   socket.on('agent:node_complete', (data) => {
       // Mark node as complete
       updateNodeStatus(data.node, 'complete');
   });
   
   socket.on('agent:complete', (data) => {
       // Final result received
       handleFinalResult(data.result);
   });
   
   socket.on('agent:error', (data) => {
       showError(data.error);
   });
   ```

3. **Update sendQuery() function:**
   ```javascript
   function sendQuery() {
       const query = inputEl.value.trim();
       if (!query) return;
       
       // Clear previous streaming state
       accumulatedText = {
           planner: '',
           evaluator: '',
           end_response: '',
           end_prompt_monitor: ''
       };
       
       // Show user message
       appendMessage('user', query);
       
       // Create placeholder for agent response
       const responseId = 'response-' + Date.now();
       appendMessage('agent', '', responseId);
       
       // Emit query via WebSocket
       socket.emit('agent:query', {
           query: query,
           session_id: sessionId,
           token: getToken()
       });
   }
   ```

4. **Add streaming text appending:**
   ```javascript
   function appendToResponse(chunk) {
       const responseEl = document.getElementById(currentResponseId);
       if (responseEl) {
           const markdownEl = responseEl.querySelector('.markdown-body');
           if (markdownEl) {
               // Append chunk and re-render markdown
               markdownEl.textContent += chunk;
               // Re-render markdown (if using marked.js)
               if (typeof marked !== 'undefined') {
                   markdownEl.innerHTML = marked.parse(markdownEl.textContent);
               }
           }
       }
   }
   ```

5. **Add visual indicators:**
   - Show typing indicator while streaming
   - Display node status (Planning → Executing → Evaluating → Generating Response)
   - Show progress bar or spinner

---

### Phase 6: Configuration and Feature Flags

**File:** `config/agent/queryagent_planner.json`

**Add streaming configuration:**
```json
{
  "streaming": {
    "enabled": true,
    "chunk_size": 1,  // Emit every N tokens (1 = every token)
    "nodes": {
      "planner": true,
      "evaluator": true,
      "end_response": true,
      "end_prompt_monitor": true
    }
  }
}
```

**File:** `agent/config.py`

**Add streaming config getter:**
```python
def is_streaming_enabled(self) -> bool:
    return self.get_nested("streaming", "enabled", default=False)

def get_streaming_nodes(self) -> Dict[str, bool]:
    return self.get_nested("streaming", "nodes", default={})
```

---

## Implementation Order

1. **Phase 1**: LLMClient streaming (foundation)
2. **Phase 2**: StreamingAgent wrapper (orchestration)
3. **Phase 3**: Modify graph nodes (integration)
4. **Phase 4**: WebSocket handler (communication)
5. **Phase 5**: UI updates (user experience)
6. **Phase 6**: Configuration (control)

## Testing Strategy

### Unit Tests
- Test `LLMClient.stream()` with mock callback
- Test streaming vs non-streaming response equality
- Test error handling during streaming

### Integration Tests
- Test full agent flow with streaming enabled
- Test WebSocket events are emitted correctly
- Test multiple concurrent sessions

### UI Tests
- Test streaming display in browser
- Test markdown rendering during streaming
- Test error handling in UI

## Performance Considerations

1. **Chunk Size**: Emit every token (chunk_size=1) for smoothest UX, but can batch for performance
2. **WebSocket Overhead**: Minimal - text chunks are small
3. **State Updates**: Emit state updates after nodes, not during streaming
4. **Concurrent Sessions**: Use SocketIO rooms to isolate sessions

## Backward Compatibility

- **Non-streaming mode**: Default behavior unchanged
- **HTTP endpoint**: Still works, returns complete response
- **WebSocket**: Optional - UI can fall back to HTTP if WebSocket unavailable
- **Configuration**: Streaming disabled by default

## Future Enhancements

1. **Streaming SQL execution**: Show query progress (if DuckDB supports it)
2. **Streaming table rendering**: Render table rows as they arrive
3. **Streaming clarification questions**: Show questions as they're generated
4. **Progress indicators**: Show estimated time remaining
5. **Cancel streaming**: Allow user to cancel mid-stream

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| WebSocket connection drops | Fall back to HTTP polling, reconnect automatically |
| Streaming breaks markdown rendering | Buffer chunks, render periodically |
| Performance impact | Make streaming optional, batch chunks if needed |
| State consistency | Emit state updates after nodes complete |
| Concurrent session conflicts | Use SocketIO rooms for isolation |

## Success Criteria

1. ✅ LLM responses stream in real-time to UI
2. ✅ User sees text appearing as it's generated
3. ✅ Multiple nodes can stream (planner, evaluator, end)
4. ✅ Non-streaming mode still works
5. ✅ WebSocket handles connection drops gracefully
6. ✅ UI shows node progress indicators
7. ✅ Final result matches non-streaming version

