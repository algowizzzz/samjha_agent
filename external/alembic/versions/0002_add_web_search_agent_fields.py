"""add web search agent fields

Revision ID: 0002_add_web_search_agent_fields
Revises: 0001_initial
Create Date: 2025-01-15
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_add_web_search_agent_fields"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IMPORTANT (Postgres): A failed ALTER TABLE inside a transaction aborts the transaction,
    # even if the Python exception is caught. Use IF NOT EXISTS to keep migrations idempotent.
    op.execute(sa.text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS tavily_api_key TEXT"))
    op.execute(sa.text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS search_scope_allowed_domains JSON"))
    op.execute(sa.text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS search_scope_blocked_domains JSON"))
    op.execute(sa.text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS default_research_depth VARCHAR(32)"))
    op.execute(sa.text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS domain_content TEXT"))
    op.execute(sa.text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS model VARCHAR(128)"))


def downgrade() -> None:
    # Remove web search specific fields
    op.drop_column("agents", "default_research_depth")
    op.drop_column("agents", "search_scope_blocked_domains")
    op.drop_column("agents", "search_scope_allowed_domains")
    op.drop_column("agents", "tavily_api_key")

