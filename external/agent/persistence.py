import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update, delete, func

from core.db.session import get_engine, get_session_factory
from core.db.models import (
    Base,
    Agent,
    AgentPrompt,
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
        },
        # Web Search Prompts
        "web_research_decider": {
            "display_name": "Web Research Decider",
            "description": "Main decision-making prompt for web research agents. Determines if a research query can be executed, what information is needed, and generates the research specification with Tavily tool plan. Used in the Web Research Decider component."
        },
        "web_research_synthesis": {
            "display_name": "Web Research Synthesis",
            "description": "Synthesizes final answers from EvidencePack (sources, claims, conflicts). Generates comprehensive research reports with citations. Used after evidence collection is complete."
        },
        "web_research_claim_extraction": {
            "display_name": "Claim Extraction",
            "description": "Extracts structured claims from raw source content (snippets, titles). Links claims to source URLs and categorizes them. Used in the Executor's claim extraction step."
        },
        "web_research_conflict_detection": {
            "display_name": "Conflict Detection",
            "description": "Detects conflicts (contradictory claims) between different sources. Assesses conflict severity and resolution status. Used in the Executor's conflict detection step."
        },
        "web_research_ask_user_clarification": {
            "display_name": "Web Research User Clarification",
            "description": "Generates helpful clarification questions for web research agents when more information is needed from the user. Used when the Web Research Decider sends ASK_USER actions."
        },
        "web_research_response_commentary": {
            "display_name": "Web Research Response Commentary",
            "description": "Generates natural language explanations of research findings for the user. Used in the final response generation step for web research agents."
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


def get_prompt_content(db, name: str, category: Optional[str] = None, agent_id: Optional[str] = None) -> Optional[str]:
    """
    Get prompt content by name, optionally filtered by category and agent_id.
    Priority: agent override > global prompt > None
    """
    # 1. Try agent-specific override first
    if agent_id:
        agent_prompt = db.get(AgentPrompt, (agent_id, name))
        if agent_prompt and agent_prompt.is_active:
            return agent_prompt.content
    
    # 2. Try global prompt
    p = db.get(Prompt, name)
    if p:
        # If category specified, verify it matches
        if category and p.category != category:
            return None
        return p.current_content
    return None


def get_agent_prompt(db, agent_id: str, prompt_name: str) -> Optional[AgentPrompt]:
    """Get agent-specific prompt override."""
    return db.get(AgentPrompt, (agent_id, prompt_name))


def list_agent_prompts(db, agent_id: str) -> List[Dict[str, Any]]:
    """List all prompts for an agent, showing which are overridden."""
    stmt = select(AgentPrompt).where(AgentPrompt.agent_id == agent_id)
    agent_prompts = {ap.prompt_name: ap for ap in db.execute(stmt).scalars().all()}
    
    # Get all prompts for the agent's category
    agent = db.get(Agent, agent_id)
    if not agent:
        return []
    
    category = "web_search" if agent.agent_type == "external" else "structured"
    all_prompts = list_prompts(db, category=category)
    
    result = []
    for prompt in all_prompts:
        agent_prompt = agent_prompts.get(prompt["name"])
        result.append({
            "name": prompt["name"],
            "display_name": prompt["display_name"],
            "description": prompt["description"],
            "category": prompt["category"],
            "is_overridden": agent_prompt is not None and agent_prompt.is_active,
            "override_content": agent_prompt.content if agent_prompt else None,
            "default_content": get_prompt_content(db, prompt["name"], category=category)
        })
    
    return result


def upsert_agent_prompt(
    db,
    agent_id: str,
    prompt_name: str,
    content: str,
    is_active: bool = True
) -> None:
    """Create or update agent-specific prompt override."""
    ap = db.get(AgentPrompt, (agent_id, prompt_name))
    if ap is None:
        ap = AgentPrompt(agent_id=agent_id, prompt_name=prompt_name, content=content, is_active=is_active)
        db.add(ap)
    else:
        ap.content = content
        ap.is_active = is_active
        ap.updated_at = datetime.utcnow()


def delete_agent_prompt(db, agent_id: str, prompt_name: str) -> None:
    """Delete agent-specific prompt override (revert to default)."""
    ap = db.get(AgentPrompt, (agent_id, prompt_name))
    if ap:
        db.delete(ap)


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


def list_conversations_for_agent(db, agent_id: str, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    List conversations for a specific agent instance and user.
    
    Returns list of conversations with:
    - id: conversation ID
    - title: first 20 characters of first user message
    - created_at: conversation creation timestamp
    - message_count: number of messages in conversation
    """
    # Query conversations filtered by agent_id and user_id
    stmt = (
        select(Conversation)
        .where(Conversation.agent_id == agent_id)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    conversations = db.execute(stmt).scalars().all()
    
    result = []
    for conv in conversations:
        # Get first user message for title
        first_msg_stmt = (
            select(Message)
            .where(Message.conversation_id == conv.id)
            .where(Message.role == "user")
            .order_by(Message.created_at.asc())
            .limit(1)
        )
        first_msg = db.execute(first_msg_stmt).scalar_one_or_none()
        title = ""
        if first_msg:
            title = first_msg.content[:20] if len(first_msg.content) > 20 else first_msg.content
        
        # Count total messages
        msg_count_stmt = (
            select(func.count(Message.id))
            .where(Message.conversation_id == conv.id)
        )
        msg_count = db.execute(msg_count_stmt).scalar() or 0
        
        result.append({
            "id": conv.id,
            "title": title,
            "created_at": conv.created_at.isoformat() if conv.created_at else "",
            "message_count": msg_count
        })
    
    return result


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
            "model": r.model or "claude-3-sonnet-20240229",  # Default to Sonnet
            "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat(),
        }
        for r in rows
    ]


def get_agent_db(db, agent_id: str) -> Optional[Dict[str, Any]]:
    a = db.get(Agent, agent_id)
    if not a:
        return None
    result = {
        "id": a.id,
        "name": a.name,
        "agent_type": a.agent_type,
        "description": a.description,
        "domain_file": a.domain_file,
        "domain_content": a.domain_content,  # Include domain content
        "data_folder": a.data_folder,
        "model": a.model or "claude-3-sonnet-20240229",  # Default to Sonnet
        "created_at": a.created_at.isoformat(),
        "updated_at": a.updated_at.isoformat(),
    }
    # Add web search specific fields if they exist (will be added via migration)
    if hasattr(a, 'tavily_api_key'):
        result["tavily_api_key"] = a.tavily_api_key
    if hasattr(a, 'search_scope_allowed_domains'):
        result["search_scope_allowed_domains"] = a.search_scope_allowed_domains
    if hasattr(a, 'search_scope_blocked_domains'):
        result["search_scope_blocked_domains"] = a.search_scope_blocked_domains
    if hasattr(a, 'default_research_depth'):
        result["default_research_depth"] = a.default_research_depth
    return result


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
    tavily_api_key: Optional[str] = None,
    search_scope_allowed_domains: Optional[list] = None,
    search_scope_blocked_domains: Optional[list] = None,
    default_research_depth: Optional[str] = None,
) -> Agent:
    # Default to Sonnet if not specified (enables thinking/reasoning)
    if model is None:
        model = "claude-3-sonnet-20240229"
    
    import json
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
    # Add web search specific fields if they exist (will be added via migration)
    if hasattr(a, 'tavily_api_key') and tavily_api_key:
        a.tavily_api_key = tavily_api_key
    if hasattr(a, 'search_scope_allowed_domains') and search_scope_allowed_domains:
        a.search_scope_allowed_domains = json.dumps(search_scope_allowed_domains) if isinstance(search_scope_allowed_domains, list) else search_scope_allowed_domains
    if hasattr(a, 'search_scope_blocked_domains') and search_scope_blocked_domains:
        a.search_scope_blocked_domains = json.dumps(search_scope_blocked_domains) if isinstance(search_scope_blocked_domains, list) else search_scope_blocked_domains
    if hasattr(a, 'default_research_depth') and default_research_depth:
        a.default_research_depth = default_research_depth
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
        # Detect category based on filename prefix
        if name.startswith("web_research_"):
            category = "web_search"
        else:
            category = "structured"  # Default for structured agents
        p = db.get(Prompt, name)
        if p is None:
            p = Prompt(name=name, category=category, current_content=content)
            db.add(p)
            imported += 1
            logger.info(f"Imported prompt: {name} (category: {category})")
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



