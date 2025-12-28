"""
Agent Run Routes with SSE Streaming

Endpoints:
- POST /api/agents/<agent_id>/runs - Start a run
- GET /api/runs/<run_id>/events - SSE stream of events
- GET /api/runs/<run_id> - Get final run state (replay)
- POST /api/runs/<run_id>/cancel - Cancel a running run
"""

import logging
import json
import threading
import queue
from typing import Dict, Any, Optional
from flask import request, jsonify, Response, stream_with_context

from routes.base_routes import BaseRoutes
from core.db.session import get_db_session
from external.agent.persistence import (
    create_conversation,
    get_or_create_conversation,
    append_message,
    create_run,
    append_event,
    list_events,
    finish_run_success,
    finish_run_error,
    save_full_results,
    get_agent_db,
)
from external.agent.agent_registry import get_agent
from external.agent.parquet_agent import handle_query

logger = logging.getLogger(__name__)


# In-memory event queues for SSE (run_id -> queue)
_event_queues: Dict[str, queue.Queue] = {}
_event_queues_lock = threading.Lock()


def _get_event_queue(run_id: str) -> queue.Queue:
    """Get or create event queue for a run"""
    with _event_queues_lock:
        if run_id not in _event_queues:
            _event_queues[run_id] = queue.Queue()
        return _event_queues[run_id]


def _emit_event(run_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Emit an event to the queue and persist to DB"""
    event_data = {
        "event_type": event_type,
        "payload": payload,
    }
    
    # Persist to DB
    try:
        with get_db_session() as db:
            seq = append_event(db, run_id, event_type, payload)
            db.commit()
            event_data["seq"] = seq
    except Exception as e:
        logger.error(f"Failed to persist event for run {run_id}: {e}")
    
    # Emit to queue
    try:
        q = _get_event_queue(run_id)
        q.put(event_data, timeout=1.0)
    except queue.Full:
        logger.warning(f"Event queue full for run {run_id}, dropping event")


class AgentRunRoutes(BaseRoutes):
    """Agent run routes with SSE streaming"""

    def __init__(self, auth_manager, tools_registry):
        super().__init__(auth_manager, tools_registry)

    def register_routes(self, app):
        """Register agent run routes"""

        @app.route('/api/agents/<agent_id>/runs', methods=['POST'])
        @self.login_required
        def start_run(agent_id):
            """Start a new agent run"""
            try:
                user_session = self.get_user_session()
                user_id = user_session.get('user_id')
                
                data = request.get_json() or {}
                user_query = data.get('query', '').strip()
                conversation_id = data.get('conversation_id')  # Optional
                show_thinking = data.get('show_thinking', False)
                model = data.get('model')  # Optional, will use default if not provided
                
                if not user_query:
                    return jsonify({"error": "Missing 'query' in request body"}), 400
                
                # Validate agent exists
                with get_db_session() as db:
                    agent = get_agent_db(db, agent_id)
                    if not agent:
                        # Fallback to file-based registry
                        agent_dict = get_agent(agent_id)
                        if not agent_dict:
                            return jsonify({"error": "Agent not found"}), 404
                        agent = {
                            "id": agent_dict.get("id"),
                            "name": agent_dict.get("name"),
                            "agent_type": agent_dict.get("agent_type"),
                            "data_folder": agent_dict.get("data_folder"),
                        }
                
                # Create or use conversation
                with get_db_session() as db:
                    conversation_id = get_or_create_conversation(db, conversation_id, user_id, agent_id)
                    
                    # Append user message
                    append_message(db, conversation_id, "user", user_query)
                    
                    # Create run
                    run_id = create_run(
                        db,
                        conversation_id=conversation_id,
                        agent_id=agent_id,
                        user_query=user_query,
                        model=model,
                        show_thinking=show_thinking,
                    )
                    db.commit()
                
                # Start run in background thread
                def run_agent():
                    try:
                        _run_agent_async(run_id, agent_id, user_query, conversation_id, show_thinking, model)
                    except Exception as e:
                        logger.error(f"Error in agent run {run_id}: {e}", exc_info=True)
                        with get_db_session() as db:
                            finish_run_error(db, run_id, "internal_error", str(e))
                            db.commit()
                        _emit_event(run_id, "run_failed", {"error": str(e)})
                
                thread = threading.Thread(target=run_agent, daemon=True)
                thread.start()
                
                return jsonify({
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "status": "running"
                })
            except Exception as e:
                logger.error(f"Error starting run: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500

        @app.route('/api/runs/<run_id>/events', methods=['GET'])
        @self.login_required
        def stream_events(run_id):
            """SSE stream of events for a run"""
            after_seq = int(request.args.get('after_seq', 0))
            
            def generate():
                # First, send any existing events (replay)
                try:
                    with get_db_session() as db:
                        existing = list_events(db, run_id, after_seq=after_seq)
                        for evt in existing:
                            yield f"data: {json.dumps(evt)}\n\n"
                            after_seq = evt.get('seq', after_seq)
                except Exception as e:
                    logger.error(f"Error replaying events for {run_id}: {e}")
                
                # Then stream new events
                q = _get_event_queue(run_id)
                while True:
                    try:
                        event_data = q.get(timeout=30.0)
                        yield f"data: {json.dumps(event_data)}\n\n"
                        if event_data.get("event_type") in ("run_completed", "run_failed"):
                            break
                    except queue.Empty:
                        # Send keepalive
                        yield ": keepalive\n\n"
                    except Exception as e:
                        logger.error(f"Error in SSE stream for {run_id}: {e}")
                        break
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                }
            )

        @app.route('/api/runs/<run_id>', methods=['GET'])
        @self.login_required
        def get_run(run_id):
            """Get final run state (for replay)"""
            try:
                from core.db.models import Run, RunResult
                with get_db_session() as db:
                    run = db.get(Run, run_id)
                    if not run:
                        return jsonify({"error": "Run not found"}), 404
                    
                    result_data = None
                    if run.results:
                        result_data = {
                            "schema": run.results.schema_json,
                            "rows": run.results.rows_json,
                        }
                    
                    return jsonify({
                        "run_id": run.id,
                        "status": run.status,
                        "user_query": run.user_query,
                        "final_sql": run.final_sql,
                        "result_summary": run.result_summary,
                        "error_type": run.error_type,
                        "created_at": run.created_at.isoformat(),
                        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                        "results": result_data,
                    })
            except Exception as e:
                logger.error(f"Error getting run {run_id}: {e}")
                return jsonify({"error": str(e)}), 500

        @app.route('/api/runs/<run_id>/cancel', methods=['POST'])
        @self.login_required
        def cancel_run(run_id):
            """Cancel a running run"""
            # TODO: Implement cancellation flag checking in agent loop
            # For now, just mark as cancelled in DB
            try:
                from core.db.models import Run
                with get_db_session() as db:
                    run = db.get(Run, run_id)
                    if not run:
                        return jsonify({"error": "Run not found"}), 404
                    if run.status != "running":
                        return jsonify({"error": "Run is not running"}), 400
                    run.status = "cancelled"
                    db.commit()
                _emit_event(run_id, "run_cancelled", {})
                return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error cancelling run {run_id}: {e}")
                return jsonify({"error": str(e)}), 500


def _is_run_cancelled(run_id: str) -> bool:
    """Check if run is cancelled"""
    try:
        from core.db.models import Run
        with get_db_session() as db:
            run = db.get(Run, run_id)
            return run and run.status == "cancelled"
    except Exception:
        return False


def _run_agent_async(
    run_id: str,
    agent_id: str,
    user_query: str,
    conversation_id: str,
    show_thinking: bool,
    model: Optional[str],
):
    """Run agent in background and emit events"""
    _emit_event(run_id, "run_started", {"query": user_query})
    
    # Check cancellation before starting
    if _is_run_cancelled(run_id):
        _emit_event(run_id, "run_cancelled", {})
        return
    
    try:
        # Load conversation history
        from core.db.models import Message
        from sqlalchemy import select
        with get_db_session() as db:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
            messages = db.execute(stmt).scalars().all()
            conversation_history = []
            for msg in messages:
                if msg.role == "user":
                    conversation_history.append({"query": msg.content})
                elif msg.role == "agent":
                    # Try to extract SQL and response from previous runs
                    conversation_history.append({"response": msg.content})
        
        # Call agent handler (with cancellation check wrapper)
        # Note: Full event instrumentation would require modifying handle_query itself
        # For now, we emit high-level events around the call
        _emit_event(run_id, "decider_done", {})
        
        if _is_run_cancelled(run_id):
            _emit_event(run_id, "run_cancelled", {})
            with get_db_session() as db:
                finish_run_error(db, run_id, "cancelled", "Run was cancelled by user")
                db.commit()
            return
        
        result = handle_query(
            user_query=user_query,
            conversation_history=conversation_history,
            prior_state=None,
            tools_registry=None,  # Will use default
            policy_limits=None,
            show_thinking=show_thinking,
            agent_id=agent_id,
        )
        
        if _is_run_cancelled(run_id):
            _emit_event(run_id, "run_cancelled", {})
            with get_db_session() as db:
                finish_run_error(db, run_id, "cancelled", "Run was cancelled by user")
                db.commit()
            return
        
        # Emit events based on result
        status = result.get("status")
        
        if status == "SUCCESS":
            final_sql = result.get("final_sql", "")
            # Use finished_output (LLM commentary) if available, fallback to result_summary
            finished_output = result.get("finished_output", "")
            result_summary = result.get("result_summary", "")
            # Prefer finished_output (LLM-generated commentary) over result_summary (basic string)
            response_text = finished_output if finished_output else result_summary
            results = result.get("results", {})
            
            # Save results to DB
            # Transform results from executor format to DB format
            # Executor format: {"columns": [...], "rows_preview": [...]}
            # DB format: {"schema": {col_name: type, ...}, "rows": [...]}
            columns = results.get("columns", [])
            rows_preview = results.get("rows_preview", [])
            
            # Build schema dict: map column names to types (default to "string")
            schema = {col: "string" for col in columns}
            rows = rows_preview
            
            with get_db_session() as db:
                save_full_results(db, run_id, schema, rows)
                # Store finished_output (LLM commentary) in result_summary field
                finish_run_success(db, run_id, final_sql, response_text)
                db.commit()
            
            # Append agent message with LLM commentary
            with get_db_session() as db:
                append_message(db, conversation_id, "agent", response_text)
                db.commit()
            
            _emit_event(run_id, "sql_generated", {"sql": final_sql})
            _emit_event(run_id, "results_ready", {"row_count": len(rows)})
            # Send finished_output (LLM commentary) to frontend
            _emit_event(run_id, "final_response", {"response": response_text})
            _emit_event(run_id, "run_completed", {"status": "success"})
            
        elif status == "ASK_USER":
            _emit_event(run_id, "ask_user", result)
            with get_db_session() as db:
                finish_run_error(db, run_id, "ask_user", "Waiting for user input")
                db.commit()
                
        elif status == "BLOCK":
            reason = result.get("reason", "Blocked")
            with get_db_session() as db:
                finish_run_error(db, run_id, "blocked", reason)
                db.commit()
            _emit_event(run_id, "run_blocked", {"reason": reason})
            _emit_event(run_id, "run_completed", {"status": "blocked"})
            
        else:  # ERROR
            reason = result.get("reason", "Unknown error")
            with get_db_session() as db:
                finish_run_error(db, run_id, "error", reason)
                db.commit()
            _emit_event(run_id, "run_failed", {"error": reason})
            
    except Exception as e:
        logger.error(f"Error in agent run {run_id}: {e}", exc_info=True)
        with get_db_session() as db:
            finish_run_error(db, run_id, "internal_error", str(e))
            db.commit()
        _emit_event(run_id, "run_failed", {"error": str(e)})

