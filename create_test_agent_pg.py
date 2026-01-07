"""Create test agent in PostgreSQL"""
from dotenv import load_dotenv
load_dotenv()

from external.core.db.session import get_db_session
from external.agent.persistence import create_agent_db, get_agent_db

with get_db_session() as db:
    agent_id = "test_deep_research_agent"
    existing = get_agent_db(db, agent_id)
    if existing:
        print(f"Agent already exists: {agent_id}")
    else:
        agent = create_agent_db(
            db,
            agent_id=agent_id,
            name="Test Deep Research Agent",
            agent_type="deep_research",
            description="Test agent for deep research integration",
            deep_research_config={
                "research_model": "anthropic:claude-3-5-haiku-20241022",
                "research_model_max_tokens": 8192,
                "summarization_model": "anthropic:claude-3-5-haiku-20241022",
                "summarization_model_max_tokens": 4096,
                "compression_model": "anthropic:claude-3-5-haiku-20241022",
                "compression_model_max_tokens": 4096,
                "final_report_model": "anthropic:claude-3-5-haiku-20241022",
                "final_report_model_max_tokens": 8192,
                "search_api": "tavily",
                "max_researcher_iterations": 2,
                "max_concurrent_research_units": 2,
                "allow_clarification": False
            }
        )
        db.commit()
        print(f"Created agent: {agent_id}")

