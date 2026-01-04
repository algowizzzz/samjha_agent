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
    """Create a financial news web search agent for bankers."""
    print("\nCreating Financial News Research Agent for Bankers...")
    
    agent_name = "Financial News Research Agent for Bankers"
    agent_id = slugify_name(agent_name)
    
    # Read domain file if it exists
    domain_file_path = Path("financial_news_domain.md")
    if domain_file_path.exists():
        domain_content = domain_file_path.read_text(encoding="utf-8")
        print(f"  Loaded domain file: {domain_file_path}")
    else:
        # Fallback domain content
        domain_content = """# Financial News Research Agent - Domain Configuration

## Authority Domains

Primary authoritative sources for financial and regulatory information:

- **Regulatory Bodies:**
  - sec.gov (Securities and Exchange Commission)
  - federalreserve.gov (Federal Reserve System)
  - fdic.gov (Federal Deposit Insurance Corporation)
  - occ.treas.gov (Office of the Comptroller of the Currency)
  - finra.org (Financial Industry Regulatory Authority)
  - cftc.gov (Commodity Futures Trading Commission)
  - treasury.gov (U.S. Department of the Treasury)

- **Financial News & Analysis:**
  - bloomberg.com
  - reuters.com
  - wsj.com (Wall Street Journal)
  - ft.com (Financial Times)
  - marketwatch.com
  - cnbc.com
  - financialtimes.com

## Search Scope

**Allowed Domains (Priority):**
- sec.gov
- federalreserve.gov
- fdic.gov
- bloomberg.com
- reuters.com
- wsj.com
- ft.com
- marketwatch.com
- cnbc.com
- finra.org
- treasury.gov

**Blocked Domains:**
- reddit.com
- twitter.com
- facebook.com
- personal blogs
- social media platforms
- unverified financial advice sites

## Research Depth Settings

**Default Research Depth:** Standard (2 iterations, 6-20 sources)

**Quality Bar:**
- Minimum sources: 6
- Maximum sources: 20
- Minimum authority sources (regulatory/official): 2
- Source types required: ["regulatory", "news", "official"]

## Key Topics for Bankers

- Company financial performance
- Regulatory compliance issues
- SEC filings (10-K, 10-Q, 8-K)
- Enforcement actions
- Market analysis
- Credit risk indicators
- Industry trends
- Management changes
- M&A activity
- Litigation and settlements
"""
        print(f"  Using default domain content (domain file not found)")
    
    allowed_domains = [
        "sec.gov", "federalreserve.gov", "fdic.gov", "bloomberg.com",
        "reuters.com", "wsj.com", "ft.com", "marketwatch.com",
        "cnbc.com", "finra.org", "treasury.gov", "occ.treas.gov", "cftc.gov"
    ]
    
    blocked_domains = ["reddit.com", "twitter.com", "facebook.com", "x.com"]
    
    with get_db_session() as db:
        # Check if agent already exists
        existing = get_agent_db(db, agent_id)
        if existing:
            print(f"  ✅ Agent {agent_id} already exists")
            print(f"     Name: {existing.get('name')}")
            print(f"     Type: {existing.get('agent_type')}")
            return agent_id
        
        # Create agent
        print(f"  Creating new agent...")
        created_agent = create_agent_db(
            db,
            agent_id=agent_id,
            name=agent_name,
            agent_type="external",
            description="Searches financial news, regulatory filings, and official sources to provide comprehensive company intelligence for banking professionals. Focuses on SEC filings, regulatory compliance, financial performance, and risk indicators.",
            domain_file="financial_news_domain.md",
            domain_content=domain_content,
            model="claude-sonnet-4-20250514",
            tavily_api_key=None,  # Can be set later via admin panel
            search_scope_allowed_domains=allowed_domains,
            search_scope_blocked_domains=blocked_domains,
            default_research_depth="standard"
        )
        db.commit()
        db.flush()  # Ensure it's written
        
        # Verify creation
        verify = get_agent_db(db, agent_id)
        if verify:
            print(f"  ✅ Created agent: {agent_id}")
            print(f"     Name: {agent_name}")
            print(f"     Type: external")
            print(f"     Model: {verify.get('model', 'claude-sonnet-4-20250514')}")
            print(f"     Research Depth: {verify.get('default_research_depth', 'standard')}")
            print(f"     Allowed domains: {len(allowed_domains)} domains")
            print(f"     Blocked domains: {len(blocked_domains)} domains")
        else:
            print(f"  ⚠️  Agent created but verification failed")
    
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


