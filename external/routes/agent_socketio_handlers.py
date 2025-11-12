"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Agent SocketIO Handlers for SAJHA MCP Server - Moved to external/
"""

import logging
import threading
from flask_socketio import emit, join_room


class AgentSocketIOHandlers:
    """Agent-related SocketIO event handlers - moved to external/"""

    def __init__(self, socketio, auth_manager, tools_registry):
        """Initialize agent socketio handlers"""
        self.socketio = socketio
        self.auth_manager = auth_manager
        self.tools_registry = tools_registry

    def register_handlers(self):
        """Register agent SocketIO event handlers"""

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
            from external.agent.session_manager import AgentSessionManager
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
                })

