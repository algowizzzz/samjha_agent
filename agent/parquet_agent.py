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
            
            # Add user clarification to state
            state["user_clarification"] = user_clarification
            state["control"] = "replan"  # Move to replan node
            
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

            if state.get("control") != "end":
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
                # Save state as-is when waiting for user clarification
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


