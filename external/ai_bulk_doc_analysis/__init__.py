"""
AI Bulk Doc Analysis (AI at Scale) feature package.

All feature-specific UI routes, templates, and static assets should live here to
minimize changes to the core SAJHA MCP Server codebase.
"""

import os
from pathlib import Path

# CRITICAL: Load DATABASE_URL BEFORE any model imports
# This ensures _is_postgresql is correctly set in models.py
# This must happen at package import time, before any submodules are imported
database_url = os.getenv("DATABASE_URL")
if not database_url:
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env.local"
        if env_file.exists():
            load_dotenv(env_file)
            database_url = os.getenv("DATABASE_URL")
    except (ImportError, Exception):
        pass

# Set DATABASE_URL in environment so models.py can detect PostgreSQL
if database_url:
    os.environ["DATABASE_URL"] = database_url

