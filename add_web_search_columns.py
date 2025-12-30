#!/usr/bin/env python3
"""
Script to add web search columns to agents table.
This is a fallback if Alembic migration doesn't work.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from core.db.session import get_engine
from sqlalchemy import text

def add_columns():
    """Add web search columns to agents table if they don't exist."""
    engine = get_engine()
    
    with engine.connect() as conn:
        # Check and add columns
        columns_to_add = [
            ("tavily_api_key", "TEXT"),
            ("search_scope_allowed_domains", "JSON"),
            ("search_scope_blocked_domains", "JSON"),
            ("default_research_depth", "VARCHAR(32)"),
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                # Check if column exists (SQLite syntax)
                result = conn.execute(text(f"PRAGMA table_info(agents)"))
                columns = [row[1] for row in result]
                
                if col_name not in columns:
                    print(f"Adding column: {col_name}")
                    conn.execute(text(f"ALTER TABLE agents ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                else:
                    print(f"Column {col_name} already exists")
            except Exception as e:
                # Try PostgreSQL syntax if SQLite fails
                try:
                    conn.execute(text(f"ALTER TABLE agents ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                    print(f"Added column: {col_name}")
                except Exception as e2:
                    print(f"Column {col_name} might already exist or error: {e2}")
        
        print("Migration complete!")

if __name__ == "__main__":
    add_columns()

