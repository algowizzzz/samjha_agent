"""Add quick_search_config column to agents table in PostgreSQL"""
import os
from dotenv import load_dotenv
from external.core.db.session import get_engine
from sqlalchemy import text

load_dotenv()

def add_column_to_postgresql():
    engine = get_engine()
    try:
        with engine.connect() as conn:
            # Check if the column already exists to prevent errors on re-run
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='agents' AND column_name='quick_search_config'"
            )).scalar_one_or_none()

            if result is None:
                conn.execute(text('ALTER TABLE agents ADD COLUMN quick_search_config JSONB'))
                conn.commit()
                print("✓ Column 'quick_search_config' added to PostgreSQL")
            else:
                print("✓ Column 'quick_search_config' already exists in PostgreSQL")
    except Exception as e:
        print(f"✗ Error adding column to PostgreSQL: {e}")

if __name__ == "__main__":
    add_column_to_postgresql()

