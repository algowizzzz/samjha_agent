"""Import Deep Research prompts into database"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure DATABASE_URL is set
if not os.getenv('DATABASE_URL'):
    print("⚠️  DATABASE_URL not set in .env, will use default")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from external.core.db.session import get_db_session
from external.agent.persistence import upsert_prompt
from external.agent.deep_research.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    final_report_generation_prompt,
    summarize_webpage_prompt,
)

PROMPTS = [
    {
        "name": "deep_research_clarify_user",
        "category": "deep_research",
        "content": clarify_with_user_instructions,
        "display_name": "User Clarification",
        "description": "Determines if user needs clarification before starting research"
    },
    {
        "name": "deep_research_research_brief",
        "category": "deep_research",
        "content": transform_messages_into_research_topic_prompt,
        "display_name": "Research Brief Generator",
        "description": "Transforms user messages into detailed research question"
    },
    {
        "name": "deep_research_supervisor",
        "category": "deep_research",
        "content": lead_researcher_prompt,
        "display_name": "Research Supervisor",
        "description": "Main supervisor prompt for planning and delegating research tasks"
    },
    {
        "name": "deep_research_researcher",
        "category": "deep_research",
        "content": research_system_prompt,
        "display_name": "Researcher",
        "description": "Prompt for individual researchers conducting searches"
    },
    {
        "name": "deep_research_compression",
        "category": "deep_research",
        "content": compress_research_system_prompt,
        "display_name": "Research Compression",
        "description": "Compresses and cleans up research findings while preserving all information"
    },
    {
        "name": "deep_research_final_report",
        "category": "deep_research",
        "content": final_report_generation_prompt,
        "display_name": "Final Report Generation",
        "description": "Generates comprehensive final research report from all findings"
    },
    {
        "name": "deep_research_summarize_webpage",
        "category": "deep_research",
        "content": summarize_webpage_prompt,
        "display_name": "Webpage Summarization",
        "description": "Summarizes raw webpage content for downstream research agents"
    },
]

def import_prompts():
    """Import all deep research prompts into database"""
    with get_db_session() as db:
        imported = 0
        for prompt_data in PROMPTS:
            try:
                upsert_prompt(
                    db,
                    prompt_data["name"],
                    prompt_data["category"],
                    prompt_data["content"],
                    editor_user_id="system"
                )
                imported += 1
                print(f"✓ Imported: {prompt_data['name']}")
            except Exception as e:
                print(f"✗ Failed to import {prompt_data['name']}: {e}")
        
        db.commit()
        print(f"\n✅ Successfully imported {imported}/{len(PROMPTS)} prompts")

if __name__ == "__main__":
    import_prompts()

