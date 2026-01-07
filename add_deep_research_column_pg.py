"""Add deep_research_config column to PostgreSQL"""
from dotenv import load_dotenv
load_dotenv()

from external.core.db.session import get_engine
from sqlalchemy import text

engine = get_engine()
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE agents ADD COLUMN deep_research_config JSONB"))
        conn.commit()
        print("✓ Column added to PostgreSQL")
    except Exception as e:
        if "already exists" in str(e) or "duplicate" in str(e).lower():
            print("✓ Column already exists")
        else:
            print(f"✗ Error: {e}")

