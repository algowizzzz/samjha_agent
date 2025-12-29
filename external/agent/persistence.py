import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update, delete, func

from core.db.session import get_engine, get_session_factory
from core.db.models import (
    Base,
    Agent,
    Prompt,
    PromptRevision,
    Conversation,
    Message,
    Run,
    RunEvent,
    RunResult,
    ToolTrace,
)

logger = logging.getLogger(__name__)


def ensure_schema() -> None:
    """
    Dev-friendly schema creation.
    In production, prefer Alembic migrations (alembic upgrade head).
    """
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_or_create_prompt(db, name: str, category: str, content: str) -> Prompt:
    p = db.get(Prompt, name)
    if p:
        return p
    p = Prompt(name=name, category=category, current_content=content)
    db.add(p)
    return p


def upsert_prompt(db, name: str, category: str, content: str, editor_user_id: Optional[str] = None) -> None:
    p = db.get(Prompt, name)
    if p is None:
        p = Prompt(name=name, category=category, current_content=content)
        db.add(p)
    else:
        p.current_content = content
        p.category = category or p.category
    db.add(PromptRevision(prompt_name=name, content=content, editor_user_id=editor_user_id))


def list_prompts(db, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List prompts with user-friendly names and descriptions"""
    # Mapping of prompt names to user-friendly display names and descriptions
    PROMPT_METADATA = {
        "decider": {
            "display_name": "Query Decider",
            "description": "Main decision-making prompt that determines if a query can be executed, what information is needed, and generates the query specification. Used in the Decider component before execution."
        },
        "ask_user_clarification": {
            "display_name": "User Clarification",
            "description": "Generates helpful clarification questions when the agent needs more information from the user. Used when the Decider sends ASK_USER actions."
        },
        "nl_to_sql_planner": {
            "display_name": "SQL Generator",
            "description": "Converts natural language query specifications into SQL queries. Used in the Executor's SQL generation step."
        },
        "sql_plan_updater": {
            "display_name": "SQL Plan Updater",
            "description": "Applies minimal fixes to SQL queries when errors are detected. Used in the Executor's SQL patching step."
        },
        "query_result_evaluator": {
            "display_name": "Result Evaluator",
            "description": "Evaluates whether query results satisfy the user's original query. Used in the Executor's evaluation step."
        },
        "response_commentary": {
            "display_name": "Response Commentary",
            "description": "Generates natural language explanations of query results for the user. Used in the Executor's final response generation step."
        }
    }
    
    stmt = select(Prompt)
    if category:
        stmt = stmt.where(Prompt.category == category)
    rows = db.execute(stmt.order_by(Prompt.name.asc())).scalars().all()
    
    result = []
    for r in rows:
        metadata = PROMPT_METADATA.get(r.name, {})
        result.append({
            "name": r.name,
            "display_name": metadata.get("display_name", r.name.replace("_", " ").title()),
            "description": metadata.get("description", "No description available."),
            "category": r.category,
            "filename": f"{r.name}.md",
            "updated_at": r.updated_at.isoformat()
        })
    return result


def get_prompt_content(db, name: str) -> Optional[str]:
    p = db.get(Prompt, name)
    return p.current_content if p else None


def create_conversation(db, user_id: Optional[str], agent_id: Optional[str]) -> str:
    c = Conversation(user_id=user_id, agent_id=agent_id)
    db.add(c)
    db.flush()
    return c.id


def get_or_create_conversation(db, conversation_id: Optional[str], user_id: Optional[str], agent_id: Optional[str]) -> str:
    """Get existing conversation or create new one"""
    if conversation_id:
        # Check if conversation exists (only if it looks like a UUID)
        # Session IDs like "sess-xxx" are not valid UUIDs, so we'll create a new conversation
        import uuid
        try:
            # Try to parse as UUID to validate it's a real DB ID
            uuid.UUID(conversation_id)
            c = db.get(Conversation, conversation_id)
            if c:
                return conversation_id
            logger.warning(f"Conversation {conversation_id} not found in DB, creating new one")
        except (ValueError, AttributeError):
            # Not a valid UUID, treat as session ID and create new conversation
            logger.info(f"Provided conversation_id '{conversation_id}' is not a UUID, creating new conversation")
    
    # Create new conversation
    c = Conversation(user_id=user_id, agent_id=agent_id)
    db.add(c)
    db.flush()
    return c.id


def append_message(db, conversation_id: str, role: str, content: str) -> str:
    m = Message(conversation_id=conversation_id, role=role, content=content)
    db.add(m)
    db.flush()
    return m.id


def create_run(
    db,
    conversation_id: Optional[str],
    agent_id: Optional[str],
    user_query: str,
    model: Optional[str],
    show_thinking: bool,
) -> str:
    r = Run(
        conversation_id=conversation_id,
        agent_id=agent_id,
        user_query=user_query,
        model=model,
        show_thinking=show_thinking,
        status="running",
    )
    db.add(r)
    db.flush()
    return r.id


def set_run_decider_output(db, run_id: str, decider_output: Dict[str, Any]) -> None:
    r = db.get(Run, run_id)
    if r:
        r.decider_output_json = decider_output


def finish_run_success(db, run_id: str, final_sql: str, result_summary: str) -> None:
    r = db.get(Run, run_id)
    if r:
        r.status = "success"
        r.error_type = None
        r.final_sql = final_sql or ""
        r.result_summary = result_summary or ""
        r.finished_at = datetime.utcnow()


def finish_run_error(db, run_id: str, error_type: str, result_summary: str = "") -> None:
    r = db.get(Run, run_id)
    if r:
        r.status = "error"
        r.error_type = error_type
        r.result_summary = result_summary or ""
        r.finished_at = datetime.utcnow()


def next_event_seq(db, run_id: str) -> int:
    stmt = select(func.max(RunEvent.seq)).where(RunEvent.run_id == run_id)
    mx = db.execute(stmt).scalar_one_or_none()
    return int(mx or 0) + 1


def append_event(db, run_id: str, event_type: str, payload: Dict[str, Any]) -> int:
    seq = next_event_seq(db, run_id)
    e = RunEvent(run_id=run_id, seq=seq, event_type=event_type, payload_json=payload or {})
    db.add(e)
    return seq


def list_events(db, run_id: str, after_seq: int = 0) -> List[Dict[str, Any]]:
    stmt = (
        select(RunEvent)
        .where(RunEvent.run_id == run_id)
        .where(RunEvent.seq > int(after_seq))
        .order_by(RunEvent.seq.asc())
    )
    rows = db.execute(stmt).scalars().all()
    return [
        {
            "seq": r.seq,
            "event_type": r.event_type,
            "payload": r.payload_json,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


def save_full_results(db, run_id: str, schema: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    rr = db.get(RunResult, run_id)
    if rr is None:
        rr = RunResult(run_id=run_id, schema_json=schema or {}, rows_json=rows or [])
        db.add(rr)
    else:
        rr.schema_json = schema or {}
        rr.rows_json = rows or []


def append_tool_trace(
    db,
    run_id: str,
    step_idx: int,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_output: Dict[str, Any],
    status: str,
    duration_ms: Optional[int] = None,
) -> None:
    t = ToolTrace(
        run_id=run_id,
        step_idx=step_idx,
        tool_name=tool_name,
        tool_args_json=tool_args or {},
        tool_output_json=tool_output or {},
        status=status,
        duration_ms=duration_ms,
    )
    db.add(t)


# Agent CRUD (DB-backed)
def list_agents_db(db) -> List[Dict[str, Any]]:
    stmt = select(Agent).order_by(Agent.created_at.desc())
    rows = db.execute(stmt).scalars().all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "agent_type": r.agent_type,
            "description": r.description,
            "domain_file": r.domain_file,
            "domain_content": r.domain_content,  # Include domain content
            "data_folder": r.data_folder,
            "model": r.model or "claude-3-5-sonnet-20241022",  # Default to Sonnet
            "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat(),
        }
        for r in rows
    ]


def get_agent_db(db, agent_id: str) -> Optional[Dict[str, Any]]:
    a = db.get(Agent, agent_id)
    if not a:
        return None
    return {
        "id": a.id,
        "name": a.name,
        "agent_type": a.agent_type,
        "description": a.description,
        "domain_file": a.domain_file,
        "domain_content": a.domain_content,  # Include domain content
        "data_folder": a.data_folder,
        "model": a.model or "claude-3-5-sonnet-20241022",  # Default to Sonnet
        "created_at": a.created_at.isoformat(),
        "updated_at": a.updated_at.isoformat(),
    }


def create_agent_db(
    db,
    agent_id: str,
    name: str,
    agent_type: str,
    description: Optional[str] = None,
    domain_file: Optional[str] = None,
    domain_content: Optional[str] = None,
    data_folder: Optional[str] = None,
    model: Optional[str] = None,
) -> Agent:
    # Default to Sonnet if not specified (enables thinking/reasoning)
    if model is None:
        model = "claude-3-5-sonnet-20241022"
    
    a = Agent(
        id=agent_id,
        name=name,
        agent_type=agent_type,
        description=description,
        domain_file=domain_file,
        domain_content=domain_content,
        data_folder=data_folder,
        model=model,
    )
    db.add(a)
    return a


def update_agent_db(
    db,
    agent_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    domain_file: Optional[str] = None,
    domain_content: Optional[str] = None,
    data_folder: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Agent]:
    """Update an existing agent. Returns the updated agent or None if not found."""
    a = db.get(Agent, agent_id)
    if not a:
        return None
    
    if name is not None:
        a.name = name
    if description is not None:
        a.description = description
    if domain_file is not None:
        a.domain_file = domain_file
    if domain_content is not None:
        a.domain_content = domain_content
    if data_folder is not None:
        a.data_folder = data_folder
    if model is not None:
        a.model = model
    
    return a


def delete_agent_db(db, agent_id: str) -> bool:
    a = db.get(Agent, agent_id)
    if a:
        db.delete(a)
        return True
    return False


# One-time import from files to DB (idempotent)
def import_prompts_from_files(db) -> int:
    """Import prompts from external/config/prompts/*.md into DB. Idempotent."""
    from pathlib import Path
    prompts_dir = Path("external/config/prompts")
    if not prompts_dir.exists():
        return 0
    imported = 0
    for prompt_file in prompts_dir.glob("*.md"):
        name = prompt_file.stem
        content = prompt_file.read_text(encoding="utf-8", errors="replace")
        category = "structured"  # All current prompts are for structured agents
        p = db.get(Prompt, name)
        if p is None:
            p = Prompt(name=name, category=category, current_content=content)
            db.add(p)
            imported += 1
            logger.info(f"Imported prompt: {name}")
    return imported


def import_agents_from_files(db) -> int:
    """Import agents from external/config/agents/*.json into DB. Idempotent."""
    from pathlib import Path
    import json
    agents_dir = Path("external/config/agents")
    if not agents_dir.exists():
        return 0
    imported = 0
    for agent_file in agents_dir.glob("*.json"):
        try:
            cfg = json.loads(agent_file.read_text(encoding="utf-8"))
            agent_id = cfg.get("id")
            if not agent_id:
                continue
            a = db.get(Agent, agent_id)
            if a is None:
                a = Agent(
                    id=agent_id,
                    name=cfg.get("name", ""),
                    agent_type=cfg.get("agent_type", "structured"),
                    description=cfg.get("description"),
                    domain_file=cfg.get("domain_file"),
                    data_folder=cfg.get("data_folder"),
                )
                db.add(a)
                imported += 1
                logger.info(f"Imported agent: {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to import agent from {agent_file}: {e}")
    return imported



