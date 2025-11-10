import uuid
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from agent.schemas import AgentState
from agent.config import QueryAgentConfig
from agent.state_manager import AgentStateManager
from agent.graph_nodes import (
    invoke_node,
    planner_node,
    clarify_node,
    replan_node,
    execute_node,
    evaluate_node,
    end_node,
)

logger = logging.getLogger(__name__)


class ParquetQueryAgent:
    """
    Orchestrates the agent control loop with DuckDB-backed execution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = QueryAgentConfig()
        self.state_manager = AgentStateManager()
        self.max_steps = 12

    def _merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        base.update(updates)
        return base
    
    def _build_conversation_history(self, session_id, user_id, num_turns=3):
        """
        Build conversation history from state.
        Returns list of last N turns with query, SQL, response, and raw_table.
        """
        try:
            state = self.state_manager.load_session_state(session_id, user_id)
            
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
            
            logger.info(f"[ParquetAgent] Built conversation history with {len(history)} turns")
            return history
            
        except Exception as e:
            logger.error(f"[ParquetAgent] Error building conversation history: {e}")
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
                formatted += f"SQL Executed: {sql}\n"
            
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
        
        return formatted
    
    def resume_with_clarification(self, session_id: str, user_clarification: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Resume agent execution after user provides clarification"""
        logger.info(f"Resuming session {session_id} with user clarification: {user_clarification[:50]}...")
        
        try:
            # Load existing state
            state = self.state_manager.load_session_state(session_id, user_id)
            
            if not state:
                logger.error(f"Session {session_id} not found")
                return {
                    "error": "Session not found",
                    "session_id": session_id,
                    "final_output": {
                        "response": "❌ Session not found. Please start a new query.",
                        "prompt_monitor": {"error": "Session not found"}
                    }
                }
            
            # Build conversation history for resumed session
            conversation_history = self._build_conversation_history(session_id, user_id, num_turns=3)
            formatted_history = self._format_conversation_history(conversation_history)
            logger.info(f"[ParquetAgent Resume] Loaded {len(conversation_history)} conversation turns")
            
            # Add user clarification and conversation history to state
            state["user_clarification"] = user_clarification
            state["control"] = "replan"  # Move to replan node
            state["conversation_history"] = formatted_history
            state["conversation_history_raw"] = conversation_history
            
            logs = state.get("logs", [])
            logs.append({"node": "resume", "timestamp": datetime.utcnow().isoformat() + "Z", 
                        "msg": f"resuming with clarification: {user_clarification[:50]}..."})
            state["logs"] = logs
            
            # Continue execution from replan
            steps = 0
            start_all = datetime.now()
            while steps < self.max_steps:
                steps += 1
                control = state.get("control", "end")
                logger.debug(f"Session {session_id} resume step {steps}: control={control}")
                
                node_start = datetime.now()
                try:
                    if control == "replan":
                        state = self._merge(state, replan_node(state, self.cfg))
                    elif control == "execute":
                        state = self._merge(state, execute_node(state, self.cfg))
                    elif control == "evaluate":
                        state = self._merge(state, evaluate_node(state, self.cfg))
                    elif control == "clarify":
                        state = self._merge(state, clarify_node(state, self.cfg))
                        # Stop again if another clarification is needed
                        break
                    elif control == "end":
                        break
                    else:
                        logger.warning(f"Unknown control state: {control}, ending agent")
                        break
                except Exception as e:
                    logger.error(f"Node {control} failed for session {session_id}: {e}", exc_info=True)
                    state["logs"].append({"node": control, "timestamp": datetime.utcnow().isoformat() + "Z", 
                                         "msg": f"node failed: {str(e)}", "level": "error"})
                    state["control"] = "end"
                    break
                    
                # record node timing
                node_ms = (datetime.now() - node_start).total_seconds() * 1000.0
                node_name = state.get("last_node") or control
                metrics = state.get("metrics") or {}
                timings = metrics.get("node_timings_ms") or {}
                timings[node_name] = timings.get(node_name, 0.0) + node_ms
                metrics["node_timings_ms"] = timings
                state["metrics"] = metrics
                
                # persist after each step
                try:
                    self.state_manager.save_session_state(session_id, user_id, state)
                except Exception as e:
                    logger.error(f"Failed to save state for session {session_id}: {e}")

            # Build clarification dict if waiting for user
            if state.get("control") == "wait_for_user":
                clarification_questions = state.get("clarification_questions", [])
                clarify_reasoning = state.get("clarify_reasoning", [])
                clarify_prompt = state.get("clarify_prompt", "")
                
                state["clarification"] = {
                    "questions": clarification_questions,
                    "reasoning": clarify_reasoning,
                    "prompt": clarify_prompt
                }
                try:
                    self.state_manager.save_session_state(session_id, user_id, state)
                except Exception as e:
                    logger.error(f"Failed to save state while waiting for clarification: {e}")
            elif state.get("control") != "end":
                # finalize gracefully
                try:
                    state = self._merge(state, end_node(state, self.cfg))
                    self.state_manager.save_session_state(session_id, user_id, state)
                except Exception as e:
                    logger.error(f"End node failed for session {session_id}: {e}", exc_info=True)

            # total time
            metrics = state.get("metrics") or {}
            metrics["total_ms"] = (datetime.now() - start_all).total_seconds() * 1000.0
            state["metrics"] = metrics
            
            logger.info(f"Agent resumed for session {session_id}: {metrics['total_ms']}ms, {steps} steps")
            return state
            
        except Exception as e:
            logger.error(f"Fatal error resuming session {session_id}: {e}", exc_info=True)
            return {
                "session_id": session_id,
                "user_id": user_id,
                "control": "end",
                "final_output": {
                    "response": f"❌ Error resuming session: {str(e)}",
                    "prompt_monitor": {
                        "error": str(e),
                        "logs": [{"node": "resume", "timestamp": datetime.utcnow().isoformat() + "Z", 
                                 "msg": f"fatal error: {str(e)}", "level": "error"}]
                    }
                },
                "logs": [{"node": "resume", "timestamp": datetime.utcnow().isoformat() + "Z", 
                         "msg": f"fatal error: {str(e)}", "level": "error"}]
            }

    def run_query(self, query: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        logger.info(f"Starting agent query for session {sid}, user {user_id}: {query[:100]}")
        
        # ALWAYS build conversation history (no keyword detection)
        conversation_history = self._build_conversation_history(sid, user_id, num_turns=3)
        formatted_history = self._format_conversation_history(conversation_history)
        logger.info(f"[ParquetAgent] Loaded {len(conversation_history)} conversation turns")
        
        try:
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
                state = self._merge(state, invoke_node(state, self.cfg, self.state_manager))
                self.state_manager.save_session_state(sid, user_id, state)
            except Exception as e:
                logger.error(f"Invoke node failed for session {sid}: {e}", exc_info=True)
                state["logs"].append({"node": "invoke", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"invoke failed: {str(e)}", "level": "error"})
                state["control"] = "end"

            steps = 0
            start_all = datetime.now()
            while steps < self.max_steps:
                steps += 1
                control = state.get("control", "end")
                logger.debug(f"Session {sid} step {steps}: control={control}")
                
                node_start = datetime.now()
                try:
                    if control == "plan":
                        state = self._merge(state, planner_node(state, self.cfg))
                    elif control == "clarify":
                        state = self._merge(state, clarify_node(state, self.cfg))
                        # For server-driven flows, the client would later provide user_clarification and resume.
                        # Here we simulate a stop-and-return clarify prompt.
                        metrics = state.get("metrics") or {}
                        metrics["clarify_turns"] = int(metrics.get("clarify_turns", 0)) + 1
                        state["metrics"] = metrics
                        break
                    elif control == "replan":
                        state = self._merge(state, replan_node(state, self.cfg))
                    elif control == "execute":
                        state = self._merge(state, execute_node(state, self.cfg))
                    elif control == "evaluate":
                        state = self._merge(state, evaluate_node(state, self.cfg))
                    elif control == "end":
                        break
                    else:
                        logger.warning(f"Unknown control state: {control}, ending agent")
                        break
                except Exception as e:
                    logger.error(f"Node {control} failed for session {sid}: {e}", exc_info=True)
                    state["logs"].append({"node": control, "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"node failed: {str(e)}", "level": "error"})
                    state["control"] = "end"
                    break
                    
                # record node timing
                node_ms = (datetime.now() - node_start).total_seconds() * 1000.0
                node_name = state.get("last_node") or control
                metrics = state.get("metrics") or {}
                timings = metrics.get("node_timings_ms") or {}
                timings[node_name] = timings.get(node_name, 0.0) + node_ms
                metrics["node_timings_ms"] = timings
                state["metrics"] = metrics
                
                # persist after each step
                try:
                    self.state_manager.save_session_state(sid, user_id, state)
                except Exception as e:
                    logger.error(f"Failed to save state for session {sid}: {e}")

            # Always call end_node unless waiting for user
            if state.get("control") == "wait_for_user":
                # Build clarification dict for UI compatibility
                clarification_questions = state.get("clarification_questions", [])
                clarify_reasoning = state.get("clarify_reasoning", [])
                clarify_prompt = state.get("clarify_prompt", "")
                
                state["clarification"] = {
                    "questions": clarification_questions,
                    "reasoning": clarify_reasoning,
                    "prompt": clarify_prompt
                }
                
                # Save state when waiting for user clarification
                try:
                    self.state_manager.save_session_state(sid, user_id, state)
                except Exception as e:
                    logger.error(f"Failed to save state for session {sid}: {e}")
            else:
                # Call end_node to finalize output (whether control is "end" or something else)
                try:
                    state = self._merge(state, end_node(state, self.cfg))
                    self.state_manager.save_session_state(sid, user_id, state)
                except Exception as e:
                    logger.error(f"End node failed for session {sid}: {e}", exc_info=True)

            # total time
            metrics = state.get("metrics") or {}
            metrics["total_ms"] = (datetime.now() - start_all).total_seconds() * 1000.0
            state["metrics"] = metrics
            
            logger.info(f"Agent query completed for session {sid}: {metrics['total_ms']}ms, {steps} steps")
            return state
            
        except Exception as e:
            logger.error(f"Fatal error in agent for session {sid}: {e}", exc_info=True)
            # Return error state
            return {
                "user_input": query,
                "session_id": sid,
                "user_id": user_id,
                "control": "end",
                "final_output": {
                    "response": f"❌ Agent error: {str(e)}",
                    "prompt_monitor": {
                        "error": str(e),
                        "logs": [{"node": "agent", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"fatal error: {str(e)}", "level": "error"}]
                    }
                },
                "logs": [{"node": "agent", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"fatal error: {str(e)}", "level": "error"}]
            }



            metrics["total_ms"] = (datetime.now() - start_all).total_seconds() * 1000.0
            state["metrics"] = metrics
            
            logger.info(f"Agent query completed for session {sid}: {metrics['total_ms']}ms, {steps} steps")
            return state
            
        except Exception as e:
            logger.error(f"Fatal error in agent for session {sid}: {e}", exc_info=True)
            # Return error state
            return {
                "user_input": query,
                "session_id": sid,
                "user_id": user_id,
                "control": "end",
                "final_output": {
                    "response": f"❌ Agent error: {str(e)}",
                    "prompt_monitor": {
                        "error": str(e),
                        "logs": [{"node": "agent", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"fatal error: {str(e)}", "level": "error"}]
                    }
                },
                "logs": [{"node": "agent", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"fatal error: {str(e)}", "level": "error"}]
            }



            metrics["total_ms"] = (datetime.now() - start_all).total_seconds() * 1000.0
            state["metrics"] = metrics
            
            logger.info(f"Agent query completed for session {sid}: {metrics['total_ms']}ms, {steps} steps")
            return state
            
        except Exception as e:
            logger.error(f"Fatal error in agent for session {sid}: {e}", exc_info=True)
            # Return error state
            return {
                "user_input": query,
                "session_id": sid,
                "user_id": user_id,
                "control": "end",
                "final_output": {
                    "response": f"❌ Agent error: {str(e)}",
                    "prompt_monitor": {
                        "error": str(e),
                        "logs": [{"node": "agent", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"fatal error: {str(e)}", "level": "error"}]
                    }
                },
                "logs": [{"node": "agent", "timestamp": datetime.utcnow().isoformat() + "Z", "msg": f"fatal error: {str(e)}", "level": "error"}]
            }


