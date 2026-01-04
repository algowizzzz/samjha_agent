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
    set_run_decider_output,
    append_event,
    list_events,
    finish_run_success,
    finish_run_error,
    save_full_results,
    get_agent_db,
    list_conversations_for_agent,
)
from external.agent.agent_registry import get_agent
from external.agent.parquet_agent import handle_query, load_domain_md

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
                cursor_seq = after_seq
                # First, send any existing events (replay)
                try:
                    with get_db_session() as db:
                        existing = list_events(db, run_id, after_seq=cursor_seq)
                        for evt in existing:
                            yield f"data: {json.dumps(evt)}\n\n"
                            cursor_seq = evt.get('seq', cursor_seq)
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

        @app.route('/api/agents/<agent_id>/conversations', methods=['GET'])
        @self.login_required
        def list_conversations(agent_id):
            """List conversations for a specific agent instance"""
            try:
                user_session = self.get_user_session()
                user_id = user_session.get('user_id')
                
                limit = int(request.args.get('limit', 50))
                
                with get_db_session() as db:
                    conversations = list_conversations_for_agent(db, agent_id, user_id, limit)
                    return jsonify(conversations)
            except Exception as e:
                logger.error(f"Error listing conversations: {e}", exc_info=True)
                return jsonify({"error": str(e)}), 500

        @app.route('/api/conversations/<conversation_id>/messages', methods=['GET'])
        @self.login_required
        def get_conversation_messages(conversation_id):
            """Get all messages for a conversation"""
            try:
                user_session = self.get_user_session()
                user_id = user_session.get('user_id')
                
                from sqlalchemy import select
                from core.db.models import Message, Conversation
                
                with get_db_session() as db:
                    # Verify conversation exists and belongs to user
                    conv = db.get(Conversation, conversation_id)
                    if not conv:
                        return jsonify({"error": "Conversation not found"}), 404
                    if conv.user_id != user_id:
                        return jsonify({"error": "Unauthorized"}), 403
                    
                    # Get all messages
                    stmt = (
                        select(Message)
                        .where(Message.conversation_id == conversation_id)
                        .order_by(Message.created_at.asc())
                    )
                    messages = db.execute(stmt).scalars().all()
                    
                    result = [
                        {
                            "id": msg.id,
                            "role": msg.role,
                            "content": msg.content,
                            "created_at": msg.created_at.isoformat() if msg.created_at else ""
                        }
                        for msg in messages
                    ]
                    
                    return jsonify({
                        "conversation_id": conversation_id,
                        "messages": result
                    })
            except Exception as e:
                logger.error(f"Error loading conversation messages: {e}", exc_info=True)
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
        # Load agent to determine agent_type
        with get_db_session() as db:
            agent = get_agent_db(db, agent_id)
            if not agent:
                # Fallback to file-based registry
                agent_dict = get_agent(agent_id)
                if not agent_dict:
                    raise ValueError(f"Agent {agent_id} not found")
                agent = {
                    "id": agent_dict.get("id"),
                    "name": agent_dict.get("name"),
                    "agent_type": agent_dict.get("agent_type", "structured"),
                    "data_folder": agent_dict.get("data_folder"),
                }
            agent_type = agent.get("agent_type", "structured")
        
        # Load conversation history
        from core.db.models import Message, Run, RunResult
        from sqlalchemy import select
        with get_db_session() as db:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
            messages = db.execute(stmt).scalars().all()
            # Build turn-based conversation history for Decider (query/response pairs).
            # This is important for USER_ANSWER classification when the last assistant turn was ASK_USER.
            conversation_history = []
            current_turn = None
            for msg in messages:
                if msg.role == "user":
                    if current_turn:
                        conversation_history.append(current_turn)
                    current_turn = {"query": msg.content}
                elif msg.role == "agent":
                    if current_turn is None:
                        current_turn = {"query": ""}
                    current_turn["response"] = msg.content
                    if isinstance(msg.content, str) and msg.content.startswith("ASK_USER:"):
                        current_turn["status"] = "ASK_USER"
            if current_turn:
                conversation_history.append(current_turn)

            # ------------------------------------------------------------------
            # Standardized continuity packet + prior structured state
            # ------------------------------------------------------------------
            # Pre-SSE we persisted this in-memory per session_id. In SSE, each user turn is a new run,
            # so we rehydrate from DB by conversation_id.
            last_run = (
                db.query(Run)
                .filter(Run.conversation_id == conversation_id)
                .filter(Run.id != run_id)
                .order_by(Run.created_at.desc())
                .first()
            )

            prior_query_spec = {}
            prior_query_spec_status = {}
            last_run_context: Dict[str, Any] = {
                "last_action": "",
                "last_query_type": "",
                "last_sql": "",
                "last_error": "",
                "last_error_type": "",
                "last_result_summary": "",
                "last_results_schema": None,
                "last_results_preview": None,
            }
            pending_clarification: Dict[str, Any] = {
                "question": "",
                "missing_field": "",
                "candidate_columns": [],
                "fills_gap": "",  # Track which query_spec field the ASK_USER was about
            }

            # Handle prior state based on agent_type
            prior_research_spec = {}
            prior_research_spec_status = {}
            
            if last_run is not None:
                dj = last_run.decider_output_json if isinstance(last_run.decider_output_json, dict) else None
                # DIAGNOSTIC: Log what we're loading from last run
                logger.info(f"[DATA_FLOW] Last run found: {last_run.id[:8]}..., status: {last_run.status}")
                logger.info(f"[DATA_FLOW] Last run decider_output_json type: {type(dj)}")
                if isinstance(dj, dict):
                    # Check if this is a web research agent (has research_spec) or structured (has query_spec)
                    if "research_spec" in dj:
                        prior_research_spec = dj.get("research_spec") or {}
                        prior_research_spec_status = dj.get("research_spec_status") or {}
                    else:
                        prior_query_spec = dj.get("query_spec") or {}
                        prior_query_spec_status = dj.get("query_spec_status") or {}
                        # DIAGNOSTIC: Log the actual prior_query_spec content
                        start_table = prior_query_spec.get("start_table", {})
                        logger.info(f"[DATA_FLOW] prior_query_spec.start_table: {start_table}")
                        logger.info(f"[DATA_FLOW] prior_query_spec.dimensions: {prior_query_spec.get('dimensions', [])}")
                        logger.info(f"[DATA_FLOW] prior_query_spec.metrics: {prior_query_spec.get('metrics', [])[:2]}...")  # First 2
                    last_run_context["last_action"] = str(dj.get("action") or "")
                    last_run_context["last_query_type"] = str(dj.get("query_type") or "")

                    # If the last turn ended with ASK_USER, capture the question (for USER_ANSWER decisions)
                    au = dj.get("ask_user") or {}
                    logger.info(f"[DATA_FLOW] Last run ask_user object: {au}")
                    if isinstance(au, dict) and au.get("question"):
                        pending_clarification["question"] = str(au.get("question") or "")
                        # Copy fills_gap directly if available - this is the most reliable way
                        if au.get("fills_gap"):
                            pending_clarification["fills_gap"] = str(au.get("fills_gap"))
                        # Best-effort: parse missing_field from question like: instead of 'promo_code'
                        try:
                            import re
                            m = re.search(r"instead of\\s+'([^']+)'", pending_clarification["question"], flags=re.IGNORECASE)
                            if m:
                                pending_clarification["missing_field"] = m.group(1)
                        except Exception:
                            pass
                        # Best-effort: parse candidate columns from "(Candidate columns: ...)"
                        try:
                            qtxt = pending_clarification["question"]
                            if "Candidate columns:" in qtxt:
                                tail = qtxt.split("Candidate columns:", 1)[1]
                                tail = tail.split(")", 1)[0]
                                cands = [c.strip().strip('"').strip("'") for c in tail.split(",") if c.strip()]
                                pending_clarification["candidate_columns"] = cands[:10]
                        except Exception:
                            pass

                last_run_context["last_sql"] = str(last_run.final_sql or "")
                last_run_context["last_error_type"] = str(last_run.error_type or "")
                last_run_context["last_result_summary"] = str(last_run.result_summary or "")
                # NOTE: we don't currently persist raw engine error text; this remains best-effort.
                last_run_context["last_error"] = str(last_run.result_summary or "")

                # Attach last results (schema + small preview) if present
                try:
                    rr = last_run.results
                    if rr is not None:
                        last_run_context["last_results_schema"] = rr.schema_json
                        if isinstance(rr.rows_json, list):
                            last_run_context["last_results_preview"] = rr.rows_json[:20]
                except Exception:
                    pass

            # Also set pending_clarification from conversation_history if the last assistant turn was ASK_USER:
            # this catches cases where ask_user.question wasn't available in decider_output_json.
            try:
                if conversation_history:
                    # Find most recent ASK_USER turn (if any)
                    for turn in reversed(conversation_history):
                        if (turn or {}).get("status") == "ASK_USER":
                            resp = str((turn or {}).get("response") or "")
                            if resp.startswith("ASK_USER:"):
                                pending_clarification["question"] = resp[len("ASK_USER:"):].strip()
                            break
            except Exception:
                pass

            # Build continuity packet based on agent_type
            if agent_type == "external":
                continuity_packet = {
                    "prior_research_spec": prior_research_spec,
                    "prior_research_spec_status": prior_research_spec_status,
                    "conversation_history": conversation_history,
                    "last_run_context": last_run_context,
                    "pending_clarification": pending_clarification,
                }
            else:
                continuity_packet = {
                    "prior_query_spec": prior_query_spec,
                    "prior_query_spec_status": prior_query_spec_status,
                    "conversation_history": conversation_history,
                    "last_run_context": last_run_context,
                    "pending_clarification": pending_clarification,
                }
        
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
        
        # Build prior_state based on agent_type
        if agent_type == "external":
            # Web research agent - different state structure
            # Web research will initialize its own state, but we can pass continuity_packet
            prior_state = None
        else:
            # Structured agent - load agent model and data folder
            agent_model = None
            agent_data_folder = None
            with get_db_session() as db:
                if agent_id:
                    try:
                        agent = get_agent_db(db, agent_id)
                        if agent:
                            agent_model = agent.get("model") or "claude-3-sonnet-20240229"
                            agent_data_folder = agent.get("data_folder")
                    except Exception as e:
                        logger.warning(f"Error loading agent config for {agent_id}: {e}")
            
            prior_state = {
                "user_query": user_query,  # will be overwritten by initialize_controller_state
                "conversation_history": conversation_history,
                "domain_md": load_domain_md(user_query, conversation_history, agent_id=agent_id),
                "policy_limits": {
                    "max_attempts": 3,
                    "max_rows": 5000,
                    "timeout_seconds": 30,
                    "allow_cross_join": False,
                },
                "query_spec": prior_query_spec,
                "query_spec_status": prior_query_spec_status,
                "show_thinking": bool(show_thinking),
                "thinking_trace": None,
                "last_executor_report": None,
                "attempt_count": 0,
                "agent_id": agent_id,
                "agent_data_folder": agent_data_folder,  # Load from DB, not None
                "agent_model": agent_model,  # Store agent's configured model
                # Keep for prompt consumption (decider.py will include this in context)
                "continuity_packet": continuity_packet,
            }
        
        # Route to appropriate handler based on agent_type
        agent_type = agent.get("agent_type") if agent else "structured"
        
        if agent_type == "external":
            # Web research agent
            from external.agent.web_research_agent import handle_web_research_query
            # Build continuity_packet for web research
            web_continuity_packet = {
                "prior_research_spec": prior_research_spec,
                "prior_research_spec_status": prior_research_spec_status,
                "conversation_history": conversation_history,
                "last_run_context": last_run_context,
                "pending_clarification": pending_clarification,
            }
            result = handle_web_research_query(
                user_query=user_query,
                conversation_history=conversation_history,
                prior_state=None,  # Web research will initialize its own state
                tools_registry=None,  # Will use default
                policy_limits=None,
                show_thinking=show_thinking,
                agent_id=agent_id,
                on_decider_output=lambda decider_output, _state: _persist_decider_output(run_id, decider_output),
                continuity_packet=web_continuity_packet,
            )
        else:
            # Structured agent (default)
            from external.agent.parquet_agent import handle_query
            result = handle_query(
                user_query=user_query,
                conversation_history=conversation_history,
                prior_state=prior_state,
                tools_registry=None,  # Will use default
                policy_limits=None,
                show_thinking=show_thinking,
                agent_id=agent_id,
                on_decider_output=lambda decider_output, _state: _persist_decider_output(run_id, decider_output),
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
            if agent_type == "external":
                # Web research agent - different response format
                final_answer = result.get("final_answer") or result.get("result_summary", "")
                evidence_pack = result.get("evidence_pack", {})
                sources = evidence_pack.get("sources", [])
                claims = evidence_pack.get("claims", [])
                conflicts = evidence_pack.get("conflicts", [])
                
                # Save evidence_pack to DB (store in result_summary as JSON for now)
                import json
                with get_db_session() as db:
                    # Store evidence_pack summary in result_summary
                    evidence_summary = {
                        "sources_count": len(sources),
                        "claims_count": len(claims),
                        "conflicts_count": len(conflicts),
                        "final_answer": final_answer
                    }
                    finish_run_success(db, run_id, None, json.dumps(evidence_summary))  # No SQL for web research
                    db.commit()
                
                # Append agent message
                with get_db_session() as db:
                    append_message(db, conversation_id, "agent", final_answer)
                    db.commit()
                
                _emit_event(run_id, "sources_collected", {"count": len(sources)})
                _emit_event(run_id, "claims_extracted", {"count": len(claims)})
                if conflicts:
                    _emit_event(run_id, "conflicts_detected", {"count": len(conflicts)})
                _emit_event(run_id, "final_response", {"response": final_answer, "evidence_pack": evidence_pack})
                _emit_event(run_id, "run_completed", {"status": "success"})
            else:
                # Structured agent - SQL results
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
                
                # Serialize non-JSON-serializable types (Timestamp, numpy, etc.) to strings
                def serialize_value(v):
                    if v is None:
                        return None
                    if hasattr(v, 'isoformat'):  # datetime, Timestamp
                        return v.isoformat()
                    if hasattr(v, 'item'):  # numpy types
                        return v.item()
                    return v
                
                rows = [{k: serialize_value(v) for k, v in row.items()} for row in rows_preview]
                
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
                # Get the question for storage
                q = result.get("question", "") or result.get("ask_user", {}).get("question", "")
                # Store actual question (or flag) in result_summary for polling clients
                is_max_recovery = result.get("is_max_attempts_recovery", False)
                summary = f"MAX_ATTEMPTS_RECOVERY: {q}" if is_max_recovery else q
                finish_run_error(db, run_id, "ask_user", summary or "Waiting for user input")
                # Persist the clarification question into the conversation so the next user message
                # can be classified as USER_ANSWER by the Decider.
                if q:
                    append_message(db, conversation_id, "agent", f"ASK_USER: {q}")
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


def _persist_decider_output(run_id: str, decider_output: Dict[str, Any]) -> None:
    """Persist latest decider output for a run (best-effort)."""
    try:
        with get_db_session() as db:
            set_run_decider_output(db, run_id, decider_output or {})
            db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist decider_output_json for run {run_id}: {e}")

