"""
Migration: Add domain_content column to agents table
Run this once to add the domain_content column to existing databases.
"""
from core.db.session import get_engine
from sqlalchemy import text


def migrate():
    """Add domain_content column to agents table if it doesn't exist"""
    engine = get_engine()
    with engine.begin() as conn:
        # PostgreSQL doesn't support IF NOT EXISTS in ALTER TABLE ADD COLUMN
        # So we check if column exists first
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='agents' AND column_name='domain_content'
        """))
        if result.fetchone() is None:
            conn.execute(text("ALTER TABLE agents ADD COLUMN domain_content TEXT"))
            print("Added domain_content column to agents table")
        else:
            print("domain_content column already exists")
    print("Migration complete")


if __name__ == "__main__":
    migrate()

