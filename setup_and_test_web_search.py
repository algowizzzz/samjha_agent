#!/usr/bin/env python3
"""
Setup and test web search agent using Flask app context.
This script can be run while the server is running or standalone.
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

def setup_with_app_context():
    """Setup using Flask app context."""
    from web.app import create_app
    
    app, socketio = create_app()
    
    with app.app_context():
        from external.core.db.session import get_db_session
        from external.agent.persistence import (
            ensure_schema,
            import_prompts_from_files,
            create_agent_db,
            get_agent_db,
            get_prompt_content
        )
        from external.agent.agent_registry import slugify_name
        
        print("=" * 60)
        print("Web Search Agent Setup (using Flask app context)")
        print("=" * 60)
        
        # 1. Ensure schema
        print("\n1. Ensuring database schema...")
        try:
            ensure_schema()
            print("  ✓ Schema ensured")
        except Exception as e:
            print(f"  ⚠ Schema check: {e}")
        
        # 2. Import prompts
        print("\n2. Importing web search prompts...")
        try:
            with get_db_session() as db:
                imported = import_prompts_from_files(db)
                db.commit()
                print(f"  ✓ Imported {imported} prompts")
                
                # Verify web search prompts
                web_search_prompts = [
                    "web_research_decider",
                    "web_research_synthesis",
                    "web_research_claim_extraction",
                    "web_research_conflict_detection",
                    "web_research_ask_user_clarification",
                    "web_research_response_commentary"
                ]
                
                found = 0
                for prompt_name in web_search_prompts:
                    content = get_prompt_content(db, prompt_name, category="web_search")
                    if content:
                        found += 1
                        print(f"    ✓ {prompt_name}")
                    else:
                        print(f"    ✗ {prompt_name} (not found)")
                
                print(f"  Found {found}/{len(web_search_prompts)} web search prompts")
        except Exception as e:
            print(f"  ✗ Failed to import prompts: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 3. Create financial news agent
        print("\n3. Creating financial news agent...")
        agent_name = "Financial News Research Agent"
        agent_id = slugify_name(agent_name)
        
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
        
        try:
            with get_db_session() as db:
                # Check if agent exists
                existing = get_agent_db(db, agent_id)
                if existing:
                    print(f"  Agent {agent_id} already exists")
                    print(f"    Name: {existing.get('name')}")
                    print(f"    Type: {existing.get('agent_type')}")
                else:
                    # Create agent
                    create_agent_db(
                        db,
                        agent_id=agent_id,
                        name=agent_name,
                        agent_type="external",
                        description="Web research agent focused on financial news, market trends, and economic indicators",
                        domain_file="financial_news_domain.md",
                        domain_content=domain_content,
                        data_folder=None,
                        model="claude-3-sonnet-20240229",
                        tavily_api_key=None,
                        search_scope_allowed_domains=["sec.gov", "federalreserve.gov", "treasury.gov", "bloomberg.com", "reuters.com", "wsj.com", "ft.com"],
                        search_scope_blocked_domains=["reddit.com", "twitter.com", "facebook.com"],
                        default_research_depth="standard"
                    )
                    db.commit()
                    print(f"  ✓ Created agent: {agent_id}")
                    print(f"    Name: {agent_name}")
                    print(f"    Type: external")
                    print(f"    Allowed domains: sec.gov, federalreserve.gov, treasury.gov, bloomberg.com, reuters.com, wsj.com, ft.com")
        except Exception as e:
            print(f"  ✗ Failed to create agent: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 4. Test agent retrieval
        print("\n4. Testing agent retrieval...")
        try:
            with get_db_session() as db:
                agent = get_agent_db(db, agent_id)
                if agent:
                    print(f"  ✓ Agent retrieved successfully")
                    print(f"    ID: {agent.get('id')}")
                    print(f"    Name: {agent.get('name')}")
                    print(f"    Type: {agent.get('agent_type')}")
                    print(f"    Description: {agent.get('description')}")
                    print(f"    Research Depth: {agent.get('default_research_depth', 'standard')}")
                    print(f"    Allowed Domains: {agent.get('search_scope_allowed_domains')}")
                    return agent_id
                else:
                    print(f"  ✗ Agent not found")
                    return None
        except Exception as e:
            print(f"  ✗ Failed to retrieve agent: {e}")
            return None
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print(f"Agent ID: {agent_id}")
        print("=" * 60)

if __name__ == "__main__":
    setup_with_app_context()


