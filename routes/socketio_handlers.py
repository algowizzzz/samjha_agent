"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
SocketIO Handlers for SAJHA MCP Server
"""

import logging
import threading
from flask import request
from flask_socketio import emit, disconnect, join_room, leave_room


class SocketIOHandlers:
    """WebSocket event handlers"""
    
    def __init__(self, socketio, auth_manager, tools_registry, mcp_handler):
        """
        Initialize SocketIO handlers
        
        Args:
            socketio: SocketIO instance
            auth_manager: Authentication manager instance
            tools_registry: Tools registry instance
            mcp_handler: MCP handler instance
        """
        self.socketio = socketio
        self.auth_manager = auth_manager
        self.tools_registry = tools_registry
        self.mcp_handler = mcp_handler
    
    def register_handlers(self):
        """Register all SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle WebSocket connection"""
            logging.info(f"WebSocket client connected: {request.sid}")
            emit('connected', {'message': 'Connected to SAJHA MCP Server'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle WebSocket disconnection"""
            logging.info(f"WebSocket client disconnected: {request.sid}")
        
        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            """Handle WebSocket authentication"""
            token = data.get('token')
            if token:
                session_data = self.auth_manager.validate_session(token)
                if session_data:
                    emit('authenticated', {
                        'success': True,
                        'user': session_data['user_name']
                    })
                    return
            
            emit('authenticated', {'success': False, 'error': 'Invalid token'})
            disconnect()
        
        @self.socketio.on('mcp_request')
        def handle_mcp_request(data):
            """Handle MCP request over WebSocket"""
            # Validate session
            token = data.get('token')
            session_data = None
            if token:
                session_data = self.auth_manager.validate_session(token)
            
            # Get request
            request_data = data.get('request')
            if not request_data:
                emit('mcp_response', {
                    'error': 'Invalid request'
                })
                return
            
            # Handle request
            response = self.mcp_handler.handle_request(request_data, session_data)
            emit('mcp_response', response)
        
        @self.socketio.on('tool_execute')
        def handle_tool_execute(data):
            """Handle tool execution over WebSocket"""
            # Validate session
            token = data.get('token')
            if not token:
                emit('tool_result', {'error': 'Unauthorized'})
                return
            
            session_data = self.auth_manager.validate_session(token)
            if not session_data:
                emit('tool_result', {'error': 'Invalid token'})
                return
            
            # Get tool and arguments
            tool_name = data.get('tool')
            arguments = data.get('arguments', {})
            
            # Check access
            if not self.auth_manager.has_tool_access(session_data, tool_name):
                emit('tool_result', {'error': 'Access denied'})
                return
            
            # Execute tool
            try:
                tool = self.tools_registry.get_tool(tool_name)
                if not tool:
                    emit('tool_result', {'error': 'Tool not found'})
                    return
                
                result = tool.execute_with_tracking(arguments)
                emit('tool_result', {
                    'success': True,
                    'result': result
                })
            except Exception as e:
                emit('tool_result', {
                    'success': False,
                    'error': str(e)
                })
        
        @self.socketio.on('join_session')
        def handle_join_session(data):
            """Join a session room for streaming isolation"""
            session_id = data.get('session_id')
            if session_id:
                join_room(session_id)
                logging.info(f"Client {request.sid} joined session room: {session_id}")
                emit('session_joined', {'session_id': session_id})
        
        @self.socketio.on('agent:query')
        def handle_agent_query(data):
            """Handle streaming agent query"""
            # Validate session
            token = data.get('token')
            if not token:
                emit('agent:error', {'error': 'Unauthorized', 'session_id': data.get('session_id')})
                return
            
            session_data = self.auth_manager.validate_session(token)
            if not session_data:
                emit('agent:error', {'error': 'Invalid token', 'session_id': data.get('session_id')})
                return
            
            # Get query and session_id
            query = data.get('query')
            user_clarification = data.get('user_clarification')  # Check if this is a clarification
            session_id = data.get('session_id')
            user_id = session_data.get('user_id')
            
            # Get config file selections from frontend
            data_dict_file = data.get('data_dict_file')
            agent_prompts_file = data.get('agent_prompts_file')
            
            if not query:
                emit('agent:error', {'error': 'Query is required', 'session_id': session_id})
                return
            
            if not session_id:
                emit('agent:error', {'error': 'Session ID is required', 'session_id': None})
                return
            
            # Join session room if not already joined
            join_room(session_id)
            
            # Get agent tool
            try:
                agent_tool = self.tools_registry.get_tool('parquet_agent')
                if not agent_tool:
                    emit('agent:error', {'error': 'Agent tool not found', 'session_id': session_id})
                    return
                
                # Create streaming wrapper
                from external.agent.streaming_agent import StreamingAgent
                streaming_agent = StreamingAgent(
                    agent=agent_tool.agent,
                    socketio=self.socketio,
                    session_id=session_id
                )
                
                # Import session manager
                from external.agent.session_manager import AgentSessionManager
                session_manager = AgentSessionManager()
                
                # Run query with streaming in background thread
                def run_agent():
                    try:
                        # Register session for cancellation tracking
                        session_manager.register_session(session_id, threading.current_thread(), user_id)
                        
                        # Check if this is a clarification response
                        if user_clarification:
                            # Resume with clarification
                            result = streaming_agent.resume_with_clarification_streaming(
                                user_clarification=user_clarification,
                                user_id=user_id,
                                data_dict_file=data_dict_file,
                                agent_prompts_file=agent_prompts_file
                            )
                        else:
                            # Start new query
                            result = streaming_agent.run_query_streaming(
                                query=query,
                                user_id=user_id,
                                data_dict_file=data_dict_file,
                                agent_prompts_file=agent_prompts_file
                            )
                        # Final result already emitted by StreamingAgent
                    except Exception as e:
                        logging.error(f"Agent streaming error for session {session_id}: {e}", exc_info=True)
                        self.socketio.emit('agent:error', {
                            'error': str(e),
                            'session_id': session_id
                        }, room=session_id)
                    finally:
                        # Always unregister session when done
                        session_manager.unregister_session(session_id)
                
                thread = threading.Thread(target=run_agent, daemon=True)
                thread.start()
                
            except Exception as e:
                logging.error(f"Error setting up agent streaming for session {session_id}: {e}", exc_info=True)
                emit('agent:error', {
                    'error': f'Failed to start agent: {str(e)}',
                    'session_id': session_id
                })
        
        @self.socketio.on('agent:kill')
        def handle_agent_kill(data):
            """Handle kill request to cancel a running agent query"""
            # Validate session
            token = data.get('token')
            if not token:
                emit('agent:kill_error', {'error': 'Unauthorized', 'session_id': data.get('session_id')})
                return
            
            session_data = self.auth_manager.validate_session(token)
            if not session_data:
                emit('agent:kill_error', {'error': 'Invalid token', 'session_id': data.get('session_id')})
                return
            
            session_id = data.get('session_id')
            if not session_id:
                emit('agent:kill_error', {'error': 'Session ID is required'})
                return
            
            # Import session manager
            from agent.session_manager import AgentSessionManager
            session_manager = AgentSessionManager()
            
            # Cancel the session
            if session_manager.cancel_session(session_id):
                logging.info(f"Kill request received for session {session_id}")
                emit('agent:killed', {
                    'session_id': session_id,
                    'message': 'Query cancelled successfully'
                }, room=session_id)
            else:
                emit('agent:kill_error', {
                    'error': 'Session not found or already completed',
                    'session_id': session_id
                }, room=session_id)