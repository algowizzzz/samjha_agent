#!/usr/bin/env python3
"""
Script to setup web search agent:
1. Import web search prompts
2. Create a financial news agent
3. Test the setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from core.db.session import get_db_session
from external.agent.persistence import (
    ensure_schema,
    import_prompts_from_files,
    create_agent_db,
    get_agent_db,
    upsert_prompt
)
from external.agent.agent_registry import slugify_name

def import_web_search_prompts():
    """Import web search prompts to database."""
    print("Importing web search prompts...")
    
    prompts_dir = Path("external/config/prompts")
    web_search_prompts = [
        "web_research_decider.md",
        "web_research_synthesis.md",
        "web_research_claim_extraction.md",
        "web_research_conflict_detection.md",
        "web_research_ask_user_clarification.md",
        "web_research_response_commentary.md",
    ]
    
    imported = 0
    with get_db_session() as db:
        for prompt_file in web_search_prompts:
            prompt_path = prompts_dir / prompt_file
            if prompt_path.exists():
                name = prompt_path.stem
                content = prompt_path.read_text(encoding="utf-8", errors="replace")
                
                # Check if prompt exists
                from core.db.models import Prompt
                existing = db.get(Prompt, name)
                if existing:
                    print(f"  Prompt {name} already exists, updating...")
                    existing.current_content = content
                    existing.category = "web_search"
                else:
                    from core.db.models import Prompt
                    prompt = Prompt(name=name, category="web_search", current_content=content)
                    db.add(prompt)
                    print(f"  Imported {name}")
                    imported += 1
        db.commit()
    
    print(f"Imported {imported} new prompts, updated existing ones")
    return imported

def create_financial_news_agent():
    """Create a financial news web search agent."""
    print("\nCreating financial news agent...")
    
    agent_name = "Financial News Research Agent"
    agent_id = slugify_name(agent_name)
    
    # Domain content for financial news agent
    domain_content = """# Financial News Research Agent Domain Configuration

## 1. Domain Identity
- **Domain Key**: financial_news
- **Purpose**: Research and analyze financial news, market trends, and economic indicators

## 2. Authority Domains
Primary authoritative sources for financial information:
- **sec.gov** - Securities and Exchange Commission (regulatory filings, company information)
- **federalreserve.gov** - Federal Reserve (monetary policy, economic data)
- **treasury.gov** - U.S. Treasury (fiscal policy, economic reports)
- **bloomberg.com** - Bloomberg (financial news, market data)
- **reuters.com** - Reuters (financial news, market analysis)
- **wsj.com** - Wall Street Journal (financial news, market commentary)
- **ft.com** - Financial Times (global financial news)

## 3. Research Depth Settings
- **Default Research Depth**: Standard
- **Max Iterations**: 2-3
- **Min Sources**: 6
- **Max Sources**: 20
- **Search Depth**: Advanced (for comprehensive research)

## 4. Source Quality Requirements
- **High Authority**: Government sites (.gov), major financial news outlets
- **Medium Authority**: Industry publications, financial blogs
- **Low Authority**: Social media, forums (excluded by default)

## 5. Search Scope
- **Allowed Domains**: sec.gov, federalreserve.gov, treasury.gov, bloomberg.com, reuters.com, wsj.com, ft.com
- **Blocked Domains**: reddit.com, twitter.com, facebook.com
- **Time Range Default**: Last 12 months (for market trends)

## 6. Research Focus Areas
- Market analysis and trends
- Company financial performance
- Economic indicators
- Regulatory changes
- Industry analysis
- Investment research
"""
    
    with get_db_session() as db:
        # Check if agent already exists
        existing = get_agent_db(db, agent_id)
        if existing:
            print(f"  Agent {agent_id} already exists")
            return agent_id
        
        # Create agent
        create_agent_db(
            db,
            agent_id=agent_id,
            name=agent_name,
            agent_type="external",
            description="Web research agent focused on financial news, market trends, and economic indicators",
            domain_file="financial_news_domain.md",
            domain_content=domain_content,
            data_folder=None,  # Not needed for web search
            model="claude-3-sonnet-20240229",
            tavily_api_key=None,  # Can be set later via admin panel
            search_scope_allowed_domains=["sec.gov", "federalreserve.gov", "treasury.gov", "bloomberg.com", "reuters.com", "wsj.com", "ft.com"],
            search_scope_blocked_domains=["reddit.com", "twitter.com", "facebook.com"],
            default_research_depth="standard"
        )
        db.commit()
        print(f"  Created agent: {agent_id}")
        print(f"  Name: {agent_name}")
        print(f"  Type: external")
        print(f"  Allowed domains: sec.gov, federalreserve.gov, treasury.gov, bloomberg.com, reuters.com, wsj.com, ft.com")
    
    return agent_id

def test_agent_retrieval(agent_id):
    """Test retrieving the agent."""
    print(f"\nTesting agent retrieval for {agent_id}...")
    
    with get_db_session() as db:
        agent = get_agent_db(db, agent_id)
        if agent:
            print(f"  ✓ Agent found")
            print(f"    Name: {agent.get('name')}")
            print(f"    Type: {agent.get('agent_type')}")
            print(f"    Description: {agent.get('description')}")
            print(f"    Research Depth: {agent.get('default_research_depth', 'standard')}")
            return True
        else:
            print(f"  ✗ Agent not found")
            return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("Web Search Agent Setup")
    print("=" * 60)
    
    # Ensure schema
    print("\n1. Ensuring database schema...")
    try:
        ensure_schema()
        print("  ✓ Schema ensured")
    except Exception as e:
        print(f"  ⚠ Schema check failed: {e}")
    
    # Import prompts
    print("\n2. Importing web search prompts...")
    try:
        imported = import_web_search_prompts()
        print(f"  ✓ Prompts imported/updated")
    except Exception as e:
        print(f"  ✗ Failed to import prompts: {e}")
        return
    
    # Create agent
    print("\n3. Creating financial news agent...")
    try:
        agent_id = create_financial_news_agent()
        print(f"  ✓ Agent created: {agent_id}")
    except Exception as e:
        print(f"  ✗ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test retrieval
    print("\n4. Testing agent retrieval...")
    try:
        test_agent_retrieval(agent_id)
    except Exception as e:
        print(f"  ✗ Failed to retrieve agent: {e}")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"Agent ID: {agent_id}")
    print("=" * 60)

if __name__ == "__main__":
    main()

