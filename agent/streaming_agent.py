"""
Streaming Agent Wrapper
Wraps ParquetQueryAgent to provide streaming LLM responses via WebSocket.
"""
import logging
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from agent.parquet_agent import ParquetQueryAgent
from agent.schemas import AgentState

logger = logging.getLogger(__name__)


class StreamingAgent:
    """
    Wrapper around ParquetQueryAgent that supports streaming LLM responses.
    Emits SocketIO events for real-time updates.
    """
    
    def __init__(self, agent: ParquetQueryAgent, socketio, session_id: str):
        """
        Initialize StreamingAgent.
        
        Args:
            agent: ParquetQueryAgent instance
            socketio: SocketIO instance for emitting events
            session_id: Session ID for room isolation
        """
        self.agent = agent
        self.socketio = socketio
        self.session_id = session_id
        self.accumulated_text: Dict[str, str] = {
            "planner": "",
            "evaluator": "",
            "end_response": "",
            "end_prompt_monitor": ""
        }
    
    def _emit(self, event: str, data: Dict[str, Any]):
        """Emit SocketIO event to session room"""
        try:
            self.socketio.emit(event, data, room=self.session_id)
        except Exception as e:
            logger.error(f"Failed to emit {event} to session {self.session_id}: {e}")
    
    def _create_stream_callback(self, node: str) -> Callable[[str], None]:
        """
        Create a callback function for streaming chunks.
        
        Args:
            node: Node identifier (planner, evaluator, end_response, end_prompt_monitor)
        
        Returns:
            Callback function that emits chunks via SocketIO
        """
        def callback(chunk: str):
            if chunk:
                self.accumulated_text[node] += chunk
                self._emit('agent:llm_chunk', {
                    'node': node,
                    'chunk': chunk,
                    'accumulated': self.accumulated_text[node],
                    'session_id': self.session_id
                })
        return callback
    
    def _build_conversation_history(self, session_id, user_id, num_turns=3):
        """
        Build conversation history from state.
        Returns list of last N turns with query, SQL, and response.
        """
        from agent.state_manager import AgentStateManager
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            state_manager = AgentStateManager()
            state = state_manager.load_session_state(session_id, user_id)
            
            if not state:
                return []
            
            # Extract conversation history
            # State contains: user_input, plan.sql, final_output.response
            history = []
            
            # Current turn (if exists in state)
            if 'user_input' in state and 'final_output' in state:
                turn = {
                    'query': state.get('user_input', ''),
                    'sql': state.get('plan', {}).get('sql', ''),
                    'response': state.get('final_output', {}).get('response', '')
                }
                # Only include if it has meaningful content
                if turn['query'] and (turn['sql'] or turn['response']):
                    history.append(turn)
            
            # TODO: If state manager stores full conversation history in future,
            # extract last N-1 additional turns here
            # For now, we only have the most recent turn
            
            logger.info(f"[ConversationHistory] Built history with {len(history)} turns")
            return history[-num_turns:]  # Last N turns
            
        except Exception as e:
            logger.error(f"[ConversationHistory] Error building history: {e}")
            return []
    
    def _format_conversation_history(self, history):
        """
        Format conversation history for LLM prompts.
        """
        if not history:
            return "No previous conversation history."
        
        formatted = "Previous Conversation:\n"
        for i, turn in enumerate(history, 1):
            formatted += f"\nTurn {i}:\n"
            formatted += f"User Query: {turn['query']}\n"
            if turn['sql'] and turn['sql'] != "-- KNOWLEDGE QUESTION":
                formatted += f"SQL Executed: {turn['sql']}\n"
            if turn['response']:
                # Truncate long responses
                response = turn['response'][:300] + "..." if len(turn['response']) > 300 else turn['response']
                formatted += f"Agent Response: {response}\n"
        
        return formatted
    
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
        
        Args:
            query: User query string
            user_id: Optional user ID
            emit_callback: Optional callback for custom event emission
        
        Returns:
            Final agent state
        """
        # Reset accumulated text
        self.accumulated_text = {
            "planner": "",
            "evaluator": "",
            "end_response": "",
            "end_prompt_monitor": ""
        }
        
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[StreamingAgent] Query: '{query}'")
            logger.info(f"[StreamingAgent] Session ID: {self.session_id}")
            
            # ALWAYS build conversation history (no keyword detection)
            conversation_history = self._build_conversation_history(self.session_id, user_id, num_turns=3)
            formatted_history = self._format_conversation_history(conversation_history)
            
            logger.info(f"[StreamingAgent] Loaded {len(conversation_history)} conversation turns")
            
            self._emit('agent:node_start', {
                'node': 'invoke',
                'session_id': self.session_id
            })
            
            # Run the agent with streaming callbacks
            # We need to modify the agent to accept streaming callbacks
            # For now, we'll patch the nodes to use streaming
            
            # Import here to avoid circular imports
            from agent.graph_nodes import (
                invoke_node, planner_node, clarify_node, replan_node,
                execute_node, evaluate_node, end_node
            )
            from agent.config import QueryAgentConfig
            from agent.state_manager import AgentStateManager
            
            cfg = QueryAgentConfig()
            state_manager = AgentStateManager()
            
            # Initialize state
            sid = self.session_id
            state: AgentState = {
                "user_input": query,
                "user_id": user_id,
                "session_id": sid,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "docs_meta": [],
                "table_schema": {},
                "logs": [],
                "control": "plan",
                "conversation_history": formatted_history,
                "conversation_history_raw": conversation_history,
                "metrics": {
                    "node_timings_ms": {},
                    "total_ms": 0.0,
                    "clarify_turns": 0,
                    "start_time": datetime.utcnow().isoformat() + "Z",
                },
            }
            
            # Initial enrich
            try:
                self._emit('agent:node_start', {'node': 'invoke', 'session_id': sid})
                state = self.agent._merge(state, invoke_node(state, cfg, state_manager))
                state_manager.save_session_state(sid, user_id, state)
                self._emit('agent:node_complete', {'node': 'invoke', 'session_id': sid})
            except Exception as e:
                logger.error(f"Invoke node failed for session {sid}: {e}", exc_info=True)
                state["logs"].append({"node": "invoke", "timestamp": datetime.utcnow().isoformat() + "Z", 
                                    "msg": f"invoke failed: {str(e)}", "level": "error"})
                state["control"] = "end"
                self._emit('agent:error', {'error': str(e), 'node': 'invoke', 'session_id': sid})
            
            steps = 0
            start_all = datetime.now()
            max_steps = self.agent.max_steps
            
            while steps < max_steps:
                steps += 1
                control = state.get("control", "end")
                logger.debug(f"Session {sid} step {steps}: control={control}")
                
                node_start = datetime.now()
                try:
                    if control == "plan":
                        self._emit('agent:node_start', {'node': 'planner', 'session_id': sid})
                        # Create streaming callback for planner
                        stream_callback = self._create_stream_callback("planner")
                        # We'll need to pass this to planner_node - modify it to accept callback
                        state = self.agent._merge(state, planner_node(state, cfg, stream_callback=stream_callback))
                        self._emit('agent:node_complete', {'node': 'planner', 'session_id': sid})
                    elif control == "clarify":
                        self._emit('agent:node_start', {'node': 'clarify', 'session_id': sid})
                        state = self.agent._merge(state, clarify_node(state, cfg))
                        metrics = state.get("metrics") or {}
                        metrics["clarify_turns"] = int(metrics.get("clarify_turns", 0)) + 1
                        state["metrics"] = metrics
                        self._emit('agent:node_complete', {'node': 'clarify', 'session_id': sid})
                        break
                    elif control == "replan":
                        self._emit('agent:node_start', {'node': 'replan', 'session_id': sid})
                        stream_callback = self._create_stream_callback("planner")
                        state = self.agent._merge(state, replan_node(state, cfg, stream_callback=stream_callback))
                        self._emit('agent:node_complete', {'node': 'replan', 'session_id': sid})
                    elif control == "execute":
                        self._emit('agent:node_start', {'node': 'execute', 'session_id': sid})
                        state = self.agent._merge(state, execute_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'execute', 'session_id': sid})
                    elif control == "evaluate":
                        self._emit('agent:node_start', {'node': 'evaluator', 'session_id': sid})
                        stream_callback = self._create_stream_callback("evaluator")
                        state = self.agent._merge(state, evaluate_node(state, cfg, stream_callback=stream_callback))
                        self._emit('agent:node_complete', {'node': 'evaluator', 'session_id': sid})
                    elif control == "end":
                        break
                    else:
                        logger.warning(f"Unknown control state: {control}, ending agent")
                        break
                except Exception as e:
                    logger.error(f"Node {control} failed for session {sid}: {e}", exc_info=True)
                    state["logs"].append({"node": control, "timestamp": datetime.utcnow().isoformat() + "Z", 
                                         "msg": f"node failed: {str(e)}", "level": "error"})
                    state["control"] = "end"
                    self._emit('agent:error', {'error': str(e), 'node': control, 'session_id': sid})
                    break
                
                # Record node timing
                node_ms = (datetime.now() - node_start).total_seconds() * 1000.0
                node_name = state.get("last_node") or control
                metrics = state.get("metrics") or {}
                timings = metrics.get("node_timings_ms") or {}
                timings[node_name] = timings.get(node_name, 0.0) + node_ms
                metrics["node_timings_ms"] = timings
                state["metrics"] = metrics
                
                # Emit state update
                self._emit('agent:state_update', {
                    'session_id': sid,
                    'control': control,
                    'last_node': node_name
                })
                
                # Persist after each step
                try:
                    state_manager.save_session_state(sid, user_id, state)
                except Exception as e:
                    logger.error(f"Failed to save state for session {sid}: {e}")
            
            # Always call end_node unless waiting for user clarification
            if state.get("control") == "wait_for_user":
                # Save state and stop - user needs to provide clarification
                try:
                    state_manager.save_session_state(sid, user_id, state)
                    self._emit('agent:waiting_for_clarification', {
                        'session_id': sid,
                        'clarify_prompt': state.get('clarify_prompt'),
                        'clarify_questions': state.get('clarify_questions')
                    })
                except Exception as e:
                    logger.error(f"Failed to save state while waiting for clarification: {e}")
            elif state.get("control") != "end" or state.get("final_output") is None:
                try:
                    self._emit('agent:node_start', {'node': 'end', 'session_id': sid})
                    # Create separate callbacks for response and prompt_monitor
                    response_callback = self._create_stream_callback("end_response")
                    prompt_monitor_callback = self._create_stream_callback("end_prompt_monitor")
                    state = self.agent._merge(state, end_node(
                        state, cfg,
                        stream_callback_response=response_callback,
                        stream_callback_prompt_monitor=prompt_monitor_callback
                    ))
                    state_manager.save_session_state(sid, user_id, state)
                    self._emit('agent:node_complete', {'node': 'end', 'session_id': sid})
                except Exception as e:
                    logger.error(f"End node failed for session {sid}: {e}", exc_info=True)
                    self._emit('agent:error', {'error': str(e), 'node': 'end', 'session_id': sid})
            
            # Total time
            metrics = state.get("metrics") or {}
            metrics["total_ms"] = (datetime.now() - start_all).total_seconds() * 1000.0
            state["metrics"] = metrics
            
            # Emit appropriate completion event based on control state
            if state.get("control") == "wait_for_user":
                logger.info(f"Agent paused for clarification in session {sid}: {metrics['total_ms']}ms, {steps} steps")
                # Don't emit 'agent:complete' - already emitted 'agent:waiting_for_clarification'
            else:
                logger.info(f"Agent completed for session {sid}: {metrics['total_ms']}ms, {steps} steps")
                # Emit final result only when truly complete
                self._emit('agent:complete', {
                    'session_id': sid,
                    'result': state
                })
            
            return state
            
        except Exception as e:
            logger.error(f"Fatal error in streaming agent for session {self.session_id}: {e}", exc_info=True)
            self._emit('agent:error', {
                'error': str(e),
                'session_id': self.session_id
            })
            raise
    
    def resume_with_clarification_streaming(
        self,
        user_clarification: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resume agent execution with user clarification (streaming version).
        
        Loads the saved session state and continues execution from replan node.
        
        Args:
            user_clarification: User's clarification response
            user_id: Optional user ID
            
        Returns:
            Final agent state
        """
        # Reset accumulated text
        self.accumulated_text = {
            "planner": "",
            "evaluator": "",
            "end_response": "",
            "end_prompt_monitor": ""
        }
        
        sid = self.session_id
        
        try:
            # Import here to avoid circular imports
            from agent.graph_nodes import (
                replan_node, clarify_node, execute_node, evaluate_node, end_node
            )
            from agent.config import QueryAgentConfig
            from agent.state_manager import AgentStateManager
            
            cfg = QueryAgentConfig()
            state_manager = AgentStateManager()
            
            # Load existing state
            state = state_manager.load_session_state(sid, user_id)
            
            if not state:
                logger.error(f"Session {sid} not found")
                self._emit('agent:error', {
                    'error': 'Session not found',
                    'session_id': sid
                })
                return {
                    "error": "Session not found",
                    "session_id": sid
                }
            
            logger.info(f"Resuming session {sid} with clarification: {user_clarification[:50]}...")
            
            # Build conversation history for resumed session
            conversation_history = self._build_conversation_history(sid, user_id, num_turns=3)
            formatted_history = self._format_conversation_history(conversation_history)
            
            logger.info(f"[StreamingAgent Resume] Loaded {len(conversation_history)} conversation turns")
            
            # Add clarification and conversation history to state
            state["user_clarification"] = user_clarification
            state["control"] = "replan"
            state["conversation_history"] = formatted_history
            state["conversation_history_raw"] = conversation_history
            
            logs = state.get("logs", [])
            logs.append({"node": "resume", "timestamp": datetime.utcnow().isoformat() + "Z", 
                        "msg": f"resuming with clarification: {user_clarification[:50]}..."})
            state["logs"] = logs
            
            # Continue execution from replan
            steps = 0
            start_all = datetime.now()
            max_steps = self.agent.max_steps
            
            while steps < max_steps:
                steps += 1
                control = state.get("control", "end")
                logger.debug(f"Session {sid} resume step {steps}: control={control}")
                
                node_start = datetime.now()
                try:
                    if control == "replan":
                        self._emit('agent:node_start', {'node': 'replan', 'session_id': sid})
                        stream_callback = self._create_stream_callback("planner")
                        state = self.agent._merge(state, replan_node(state, cfg, stream_callback=stream_callback))
                        self._emit('agent:node_complete', {'node': 'replan', 'session_id': sid})
                    elif control == "clarify":
                        self._emit('agent:node_start', {'node': 'clarify', 'session_id': sid})
                        state = self.agent._merge(state, clarify_node(state, cfg))
                        metrics = state.get("metrics") or {}
                        metrics["clarify_turns"] = int(metrics.get("clarify_turns", 0)) + 1
                        state["metrics"] = metrics
                        self._emit('agent:node_complete', {'node': 'clarify', 'session_id': sid})
                        break
                    elif control == "execute":
                        self._emit('agent:node_start', {'node': 'execute', 'session_id': sid})
                        state = self.agent._merge(state, execute_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'execute', 'session_id': sid})
                    elif control == "evaluate":
                        self._emit('agent:node_start', {'node': 'evaluator', 'session_id': sid})
                        stream_callback = self._create_stream_callback("evaluator")
                        state = self.agent._merge(state, evaluate_node(state, cfg, stream_callback=stream_callback))
                        self._emit('agent:node_complete', {'node': 'evaluator', 'session_id': sid})
                    elif control == "end":
                        break
                    else:
                        logger.warning(f"Unknown control state: {control}, ending agent")
                        break
                except Exception as e:
                    logger.error(f"Node {control} failed for session {sid}: {e}", exc_info=True)
                    state["logs"].append({"node": control, "timestamp": datetime.utcnow().isoformat() + "Z", 
                                         "msg": f"node failed: {str(e)}", "level": "error"})
                    state["control"] = "end"
                    self._emit('agent:error', {'error': str(e), 'node': control, 'session_id': sid})
                    break
                
                # Record node timing
                node_ms = (datetime.now() - node_start).total_seconds() * 1000.0
                node_name = state.get("last_node") or control
                metrics = state.get("metrics") or {}
                timings = metrics.get("node_timings_ms") or {}
                timings[node_name] = timings.get(node_name, 0.0) + node_ms
                metrics["node_timings_ms"] = timings
                state["metrics"] = metrics
                
                # Emit state update
                self._emit('agent:state_update', {
                    'session_id': sid,
                    'control': control,
                    'last_node': node_name
                })
                
                # Persist after each step
                try:
                    state_manager.save_session_state(sid, user_id, state)
                except Exception as e:
                    logger.error(f"Failed to save state for session {sid}: {e}")
            
            # Check if waiting for clarification again
            if state.get("control") == "wait_for_user":
                # Build clarification dict
                clarification_questions = state.get("clarification_questions", [])
                clarify_reasoning = state.get("clarify_reasoning", [])
                clarify_prompt = state.get("clarify_prompt", "")
                
                state["clarification"] = {
                    "questions": clarification_questions,
                    "reasoning": clarify_reasoning,
                    "prompt": clarify_prompt
                }
                
                try:
                    state_manager.save_session_state(sid, user_id, state)
                    self._emit('agent:waiting_for_clarification', {
                        'session_id': sid,
                        'result': state
                    })
                except Exception as e:
                    logger.error(f"Failed to save state while waiting for clarification: {e}")
            elif state.get("control") != "end" or state.get("final_output") is None:
                # Call end_node
                try:
                    self._emit('agent:node_start', {'node': 'end', 'session_id': sid})
                    response_callback = self._create_stream_callback("end_response")
                    prompt_monitor_callback = self._create_stream_callback("end_prompt_monitor")
                    state = self.agent._merge(state, end_node(
                        state, cfg,
                        stream_callback_response=response_callback,
                        stream_callback_prompt_monitor=prompt_monitor_callback
                    ))
                    state_manager.save_session_state(sid, user_id, state)
                    self._emit('agent:node_complete', {'node': 'end', 'session_id': sid})
                except Exception as e:
                    logger.error(f"End node failed for session {sid}: {e}", exc_info=True)
                    self._emit('agent:error', {'error': str(e), 'node': 'end', 'session_id': sid})
            
            # Total time
            metrics = state.get("metrics") or {}
            metrics["total_ms"] = (datetime.now() - start_all).total_seconds() * 1000.0
            state["metrics"] = metrics
            
            # Emit completion
            if state.get("control") != "wait_for_user":
                self._emit('agent:complete', {
                    'session_id': sid,
                    'result': state
                })
            
            logger.info(f"Agent resumed for session {sid}: {metrics['total_ms']}ms, {steps} steps")
            return state
        
        except Exception as e:
            logger.error(f"Fatal error resuming session {sid}: {e}", exc_info=True)
            self._emit('agent:error', {
                'error': str(e),
                'session_id': sid
            })
            return {
                "session_id": sid,
                "user_id": user_id,
                "control": "end",
                "error": str(e)
            }

