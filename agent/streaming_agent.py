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
    
    def _emit_node_data(self, node_name: str, state: AgentState):
        """
        Extract and emit relevant data from a node's state.
        
        Args:
            node_name: Name of the node that just completed
            state: Current agent state after node execution
        """
        node_data = {
            'node': node_name,
            'session_id': self.session_id,
            'data': {}
        }
        
        # Extract node-specific data based on node type
        if node_name == "invoke":
            # For invoke, emit basic state info
            node_data['data'] = {
                'session_id': self.session_id,
                'has_table_schema': bool(state.get("table_schema")),
                'has_conversation_history': bool(state.get("conversation_history")),
                'control': state.get("control", "unknown")
            }
        elif node_name == "execute_sql":
            exec_result = state.get("execution_result")
            if exec_result:
                node_data['data'] = {
                    'raw_table': {
                        'columns': exec_result.get('columns', []),
                        'rows': exec_result.get('rows', []),
                        'row_count': exec_result.get('row_count', 0),
                        'query': exec_result.get('query', '')
                    },
                    'execution_stats': state.get("execution_stats", {})
                }
        elif node_name == "generate_sql":
            plan = state.get("plan", {})
            if plan:
                node_data['data'] = {
                    'plan': {
                        'sql': plan.get('sql', ''),
                        'target_table': plan.get('target_table', ''),
                        'explanation': plan.get('explanation', '')
                    },
                    'plan_quality': state.get("plan_quality", ""),
                    'plan_explain': state.get("plan_explain", "")
                }
        elif node_name == "check_followup":
            node_data['data'] = {
                'is_followup': state.get("is_followup", False),
                'conversation_history_length': len(state.get("conversation_history", [])) if isinstance(state.get("conversation_history"), list) else 0
            }
        elif node_name == "check_data_sufficiency":
            node_data['data'] = {
                'is_followup': state.get("is_followup", False),
                'more_data_needed': state.get("more_data_needed", False),
                'node_output': state.get("node_output", {})
            }
        elif node_name == "check_structure":
            node_data['data'] = {
                'is_structured': state.get("is_structured", False),
                'node_output': state.get("node_output", {})
            }
        elif node_name == "check_ambiguity":
            node_data['data'] = {
                'is_ambiguous': state.get("is_ambiguous", False),
                'clarification_questions': state.get("clarification_questions", []),
                'node_output': state.get("node_output", {})
            }
        elif node_name == "synthesize":
            # For synthesize, include the raw_table if available
            exec_result = state.get("execution_result")
            if exec_result:
                node_data['data'] = {
                    'raw_table': {
                        'columns': exec_result.get('columns', []),
                        'rows': exec_result.get('rows', []),
                        'row_count': exec_result.get('row_count', 0),
                        'query': exec_result.get('query', '')
                    }
                }
        elif node_name == "end":
            # For end node, check final_output first, then execution_result
            final_output = state.get("final_output", {})
            if final_output:
                raw_table = final_output.get("raw_table", {})
                if raw_table:
                    node_data['data'] = {
                        'raw_table': {
                            'columns': raw_table.get('columns', []),
                            'rows': raw_table.get('rows', []),
                            'row_count': raw_table.get('row_count', 0),
                            'query': raw_table.get('query', '')
                        },
                        'response': final_output.get('response', ''),
                        'prompt_monitor': final_output.get('prompt_monitor', {})
                    }
                else:
                    # final_output exists but no raw_table - emit what we have
                    node_data['data'] = {
                        'has_final_output': True,
                        'has_raw_table': False,
                        'response': final_output.get('response', ''),
                        'prompt_monitor': final_output.get('prompt_monitor', {})
                    }
            else:
                # Fallback to execution_result if final_output not available
                exec_result = state.get("execution_result")
                if exec_result:
                    node_data['data'] = {
                        'raw_table': {
                            'columns': exec_result.get('columns', []),
                            'rows': exec_result.get('rows', []),
                            'row_count': exec_result.get('row_count', 0),
                            'query': exec_result.get('query', '')
                        },
                        'has_final_output': False,
                        'source': 'execution_result'
                    }
                else:
                    # No final_output and no execution_result - emit status
                    node_data['data'] = {
                        'has_final_output': False,
                        'has_execution_result': False,
                        'control': state.get('control', 'unknown'),
                        'status': 'completed_without_output'
                    }
        
        # Always include node_output if it exists
        if state.get("node_output"):
            if 'node_output' not in node_data['data']:
                node_data['data']['node_output'] = state.get("node_output")
        
        # Always include node_reasoning if it exists
        if state.get("node_reasoning"):
            node_data['data']['node_reasoning'] = state.get("node_reasoning")
        
        # Emit the node data
        logger.info(f"[STREAMING_AGENT] Preparing node_data for {node_name}: data keys = {list(node_data['data'].keys())}")
        
        if node_data['data']:  # Only emit if there's actual data
            logger.info(f"[STREAMING_AGENT] Emitting agent:node_data for {node_name} with {len(node_data['data'])} data keys")
            try:
                self._emit('agent:node_data', node_data)
                logger.info(f"[STREAMING_AGENT] ✅ Successfully emitted agent:node_data for {node_name}")
            except Exception as e:
                logger.error(f"[STREAMING_AGENT] ❌ Failed to emit node_data for {node_name}: {e}", exc_info=True)
        else:
            logger.warning(f"[STREAMING_AGENT] ⚠️ No data to emit for {node_name} - skipping node_data event (data dict is empty)")
    
    def _create_stream_callback(self, node: str) -> Callable[[str], None]:
        """
        Create a callback function for streaming chunks.
        
        Args:
            node: Node identifier (planner, evaluator, end_response, end_prompt_monitor, check_ambiguity, generate_sql, etc.)
        
        Returns:
            Callback function that emits chunks via SocketIO
        """
        def callback(chunk: str):
            if chunk:
                # Initialize key if it doesn't exist
                if node not in self.accumulated_text:
                    self.accumulated_text[node] = ""
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
        Returns list of last N turns with query, SQL, response, and raw_table.
        """
        from agent.state_manager import AgentStateManager
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            state_manager = AgentStateManager()
            state = state_manager.load_session_state(session_id, user_id)
            
            if not state:
                return []
            
            # Extract conversation history from conversation_history_raw if available
            conversation_history_raw = state.get("conversation_history_raw", [])
            if conversation_history_raw and isinstance(conversation_history_raw, list):
                # Use the stored conversation history (includes raw_table)
                history = conversation_history_raw[-num_turns:]
            else:
                # Fallback: Build from current state
                history = []
                if 'user_input' in state and 'final_output' in state:
                    turn = {
                        'query': state.get('user_input', ''),
                        'sql': state.get('plan', {}).get('sql', ''),
                        'response': state.get('final_output', {}).get('response', ''),
                        'raw_table': state.get('raw_table') or state.get('final_output', {}).get('raw_table')
                    }
                    # Only include if it has meaningful content
                    if turn['query'] and (turn['sql'] or turn['response']):
                        history.append(turn)
            
            logger.info(f"[ConversationHistory] Built history with {len(history)} turns")
            return history
            
        except Exception as e:
            logger.error(f"[ConversationHistory] Error building history: {e}")
            return []
    
    def _format_conversation_history(self, history):
        """
        Format conversation history for LLM prompts, including table data.
        """
        if not history:
            return "No previous conversation history."
        
        formatted = "Previous Conversation:\n"
        for i, turn in enumerate(history, 1):
            formatted += f"\nTurn {i}:\n"
            formatted += f"User Query: {turn.get('query', turn.get('user_query', ''))}\n"
            
            sql = turn.get('sql') or turn.get('plan_sql') or turn.get('sql_executed', '')
            if sql and sql != "-- KNOWLEDGE QUESTION":
                formatted += f"SQL Executed:\n```sql\n{sql}\n```\n"
            
            # Include table data if available
            raw_table = turn.get('raw_table') or turn.get('execution_result')
            if raw_table:
                columns = raw_table.get('columns', [])
                rows = raw_table.get('rows', [])
                row_count = raw_table.get('row_count', len(rows))
                
                if row_count > 0 and columns:
                    formatted += f"Result Table ({row_count} row{'s' if row_count != 1 else ''}):\n"
                    formatted += f"Columns: {', '.join(columns)}\n"
                    
                    # Include ALL rows (full table data, no truncation)
                    if rows:
                        formatted += "Data:\n"
                        for j, row in enumerate(rows, 1):
                            if isinstance(row, dict):
                                row_str = " | ".join([f"{col}: {row.get(col, 'N/A')}" for col in columns])
                            else:
                                row_str = str(row)
                            formatted += f"  Row {j}: {row_str}\n"
            
            response = turn.get('response') or turn.get('response_summary', '')
            if response:
                # Include full response (no truncation)
                formatted += f"Agent Response: {response}\n"
            
            # Include prompt monitor (reasoning) if available
            prompt_monitor = turn.get('prompt_monitor')
            if prompt_monitor:
                # Handle both dict and string formats
                if isinstance(prompt_monitor, dict):
                    reasoning = prompt_monitor.get('procedural_reasoning', '') or prompt_monitor.get('reasoning', '')
                    if reasoning:
                        formatted += f"Prompt Monitor (Reasoning): {reasoning}\n"
                elif isinstance(prompt_monitor, str) and prompt_monitor.strip():
                    formatted += f"Prompt Monitor (Reasoning): {prompt_monitor}\n"
        
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
                invoke_node, check_followup_node, check_data_sufficiency_node, check_structure_node, check_ambiguity_node,
                clarify_node, process_clarification_node,
                generate_sql_node, execute_sql_node, retry_sql_node,
                synthesize_response_node, end_node
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
                "control": "invoke",
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
                self._emit_node_data('invoke', state)
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
                    if control == "invoke":
                        self._emit('agent:node_start', {'node': 'invoke', 'session_id': sid})
                        state = self.agent._merge(state, invoke_node(state, cfg, self.agent.state_manager))
                        self._emit('agent:node_complete', {'node': 'invoke', 'session_id': sid})
                        self._emit_node_data('invoke', state)
                    elif control == "check_followup":
                        self._emit('agent:node_start', {'node': 'check_followup', 'session_id': sid})
                        state = self.agent._merge(state, check_followup_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'check_followup', 'session_id': sid})
                        self._emit_node_data('check_followup', state)
                    elif control == "check_data_sufficiency":
                        self._emit('agent:node_start', {'node': 'check_data_sufficiency', 'session_id': sid})
                        state = self.agent._merge(state, check_data_sufficiency_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'check_data_sufficiency', 'session_id': sid})
                        self._emit_node_data('check_data_sufficiency', state)
                    elif control == "check_structure":
                        self._emit('agent:node_start', {'node': 'check_structure', 'session_id': sid})
                        state = self.agent._merge(state, check_structure_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'check_structure', 'session_id': sid})
                        self._emit_node_data('check_structure', state)
                    elif control == "check_ambiguity":
                        self._emit('agent:node_start', {'node': 'check_ambiguity', 'session_id': sid})
                        stream_callback = self._create_stream_callback("check_ambiguity")
                        state = self.agent._merge(state, check_ambiguity_node(state, cfg, stream_callback))
                        self._emit('agent:node_complete', {'node': 'check_ambiguity', 'session_id': sid})
                        self._emit_node_data('check_ambiguity', state)
                    elif control == "clarify":
                        self._emit('agent:node_start', {'node': 'clarify', 'session_id': sid})
                        state = self.agent._merge(state, clarify_node(state, cfg))
                        metrics = state.get("metrics") or {}
                        metrics["clarify_turns"] = int(metrics.get("clarify_turns", 0)) + 1
                        state["metrics"] = metrics
                        self._emit('agent:node_complete', {'node': 'clarify', 'session_id': sid})
                        break  # Wait for user
                    elif control == "process_clarification":
                        self._emit('agent:node_start', {'node': 'process_clarification', 'session_id': sid})
                        state = self.agent._merge(state, process_clarification_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'process_clarification', 'session_id': sid})
                    elif control == "generate_sql":
                        self._emit('agent:node_start', {'node': 'generate_sql', 'session_id': sid})
                        stream_callback = self._create_stream_callback("generate_sql")
                        state = self.agent._merge(state, generate_sql_node(state, cfg, stream_callback))
                        self._emit('agent:node_complete', {'node': 'generate_sql', 'session_id': sid})
                        self._emit_node_data('generate_sql', state)
                    elif control == "execute_sql":
                        self._emit('agent:node_start', {'node': 'execute_sql', 'session_id': sid})
                        state = self.agent._merge(state, execute_sql_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'execute_sql', 'session_id': sid})
                        self._emit_node_data('execute_sql', state)
                    elif control == "retry_sql":
                        self._emit('agent:node_start', {'node': 'retry_sql', 'session_id': sid})
                        state = self.agent._merge(state, retry_sql_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'retry_sql', 'session_id': sid})
                    elif control == "synthesize":
                        self._emit('agent:node_start', {'node': 'synthesize', 'session_id': sid})
                        stream_callback = self._create_stream_callback("end_response")
                        state = self.agent._merge(state, synthesize_response_node(state, cfg, stream_callback))
                        self._emit('agent:node_complete', {'node': 'synthesize', 'session_id': sid})
                        self._emit_node_data('synthesize', state)
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
                    self._emit_node_data('end', state)
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
                
                # Debug: Log what's being sent
                final_output = state.get("final_output", {})
                raw_table = final_output.get("raw_table", {}) if final_output else {}
                logger.info(f"[STREAMING_AGENT] Emitting agent:complete - raw_table row_count={raw_table.get('row_count', 'MISSING')}, columns={len(raw_table.get('columns', []))}, rows={len(raw_table.get('rows', []))}")
                
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
                check_followup_node, check_data_sufficiency_node, check_structure_node, check_ambiguity_node,
                clarify_node, process_clarification_node,
                generate_sql_node, execute_sql_node, retry_sql_node,
                synthesize_response_node, end_node
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
            state["control"] = "process_clarification"
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
                    if control == "process_clarification":
                        self._emit('agent:node_start', {'node': 'process_clarification', 'session_id': sid})
                        state = self.agent._merge(state, process_clarification_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'process_clarification', 'session_id': sid})
                    elif control == "check_ambiguity":
                        self._emit('agent:node_start', {'node': 'check_ambiguity', 'session_id': sid})
                        stream_callback = self._create_stream_callback("check_ambiguity")
                        state = self.agent._merge(state, check_ambiguity_node(state, cfg, stream_callback))
                        self._emit('agent:node_complete', {'node': 'check_ambiguity', 'session_id': sid})
                        self._emit_node_data('check_ambiguity', state)
                    elif control == "clarify":
                        self._emit('agent:node_start', {'node': 'clarify', 'session_id': sid})
                        state = self.agent._merge(state, clarify_node(state, cfg))
                        metrics = state.get("metrics") or {}
                        metrics["clarify_turns"] = int(metrics.get("clarify_turns", 0)) + 1
                        state["metrics"] = metrics
                        self._emit('agent:node_complete', {'node': 'clarify', 'session_id': sid})
                        break  # Wait for user
                    elif control == "generate_sql":
                        self._emit('agent:node_start', {'node': 'generate_sql', 'session_id': sid})
                        stream_callback = self._create_stream_callback("generate_sql")
                        state = self.agent._merge(state, generate_sql_node(state, cfg, stream_callback))
                        self._emit('agent:node_complete', {'node': 'generate_sql', 'session_id': sid})
                        self._emit_node_data('generate_sql', state)
                    elif control == "execute_sql":
                        self._emit('agent:node_start', {'node': 'execute_sql', 'session_id': sid})
                        state = self.agent._merge(state, execute_sql_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'execute_sql', 'session_id': sid})
                        self._emit_node_data('execute_sql', state)
                    elif control == "retry_sql":
                        self._emit('agent:node_start', {'node': 'retry_sql', 'session_id': sid})
                        state = self.agent._merge(state, retry_sql_node(state, cfg))
                        self._emit('agent:node_complete', {'node': 'retry_sql', 'session_id': sid})
                    elif control == "synthesize":
                        self._emit('agent:node_start', {'node': 'synthesize', 'session_id': sid})
                        stream_callback = self._create_stream_callback("end_response")
                        state = self.agent._merge(state, synthesize_response_node(state, cfg, stream_callback))
                        self._emit('agent:node_complete', {'node': 'synthesize', 'session_id': sid})
                        self._emit_node_data('synthesize', state)
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
                    self._emit_node_data('end', state)
                except Exception as e:
                    logger.error(f"End node failed for session {sid}: {e}", exc_info=True)
                    self._emit('agent:error', {'error': str(e), 'node': 'end', 'session_id': sid})
            
            # Total time
            metrics = state.get("metrics") or {}
            metrics["total_ms"] = (datetime.now() - start_all).total_seconds() * 1000.0
            state["metrics"] = metrics
            
            # Emit completion
            if state.get("control") != "wait_for_user":
                # Debug: Log what's being sent
                final_output = state.get("final_output", {})
                raw_table = final_output.get("raw_table", {}) if final_output else {}
                logger.info(f"[STREAMING_AGENT] Emitting agent:complete - raw_table row_count={raw_table.get('row_count', 'MISSING')}, columns={len(raw_table.get('columns', []))}, rows={len(raw_table.get('rows', []))}")
                
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

