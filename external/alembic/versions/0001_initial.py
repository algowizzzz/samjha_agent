"""initial agent persistence tables

Revision ID: 0001_initial
Revises: 
Create Date: 2025-12-28
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agents",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("agent_type", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("domain_file", sa.String(length=256), nullable=True),
        sa.Column("data_folder", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "prompts",
        sa.Column("name", sa.String(length=128), primary_key=True),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("current_content", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "prompt_revisions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("prompt_name", sa.String(length=128), sa.ForeignKey("prompts.name"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("editor_user_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("agent_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("conversation_id", sa.String(length=64), sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("conversation_id", sa.String(length=64), sa.ForeignKey("conversations.id"), nullable=True),
        sa.Column("agent_id", sa.String(length=128), nullable=True),
        sa.Column("user_query", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("show_thinking", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("decider_output_json", sa.JSON(), nullable=True),
        sa.Column("final_sql", sa.Text(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "run_events",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("run_id", "seq", name="uq_run_events_run_id_seq"),
    )

    op.create_table(
        "run_results",
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), primary_key=True),
        sa.Column("schema_json", sa.JSON(), nullable=False),
        sa.Column("rows_json", sa.JSON(), nullable=False),
    )

    op.create_table(
        "tool_traces",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("step_idx", sa.Integer(), nullable=False),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("tool_args_json", sa.JSON(), nullable=False),
        sa.Column("tool_output_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("tool_traces")
    op.drop_table("run_results")
    op.drop_table("run_events")
    op.drop_table("runs")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("prompt_revisions")
    op.drop_table("prompts")
    op.drop_table("agents")



