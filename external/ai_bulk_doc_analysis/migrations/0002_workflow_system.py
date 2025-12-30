"""
Migration: Add workflow system tables

Revision ID: 0002_workflow_system
Revises: 0001_initial (if exists, otherwise None)
Create Date: 2025-01-XX

This migration adds:
- workflows table (top-level entity)
- workflow_versions table (immutable snapshots)
- workflow_domains table (many-to-many)
- ingestion_profiles table
- export_profiles table
- execution_tasks table (for CSV row-based execution)

And modifies:
- chain_steps: add title column
- runs: add workflow_version_id column
- step_results: add task_id column
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

# revision identifiers
revision = '0002_workflow_system'
down_revision = None  # Will be set to '0001_initial' if that exists
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add workflow system tables and modify existing tables."""
    
    # Detect if using PostgreSQL
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == 'postgresql'
    
    # Use JSONB for PostgreSQL, JSON for SQLite
    json_type = JSONB().with_variant(sa.JSON(), 'sqlite')
    array_type = ARRAY(sa.String).with_variant(sa.Text(), 'sqlite')
    
    # ============================================================================
    # 1. WORKFLOWS (Top-Level Entity)
    # ============================================================================
    op.create_table(
        'workflows',
        sa.Column('workflow_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('visibility_scope', sa.String(50), nullable=False),  # 'super' | 'domain'
        sa.Column('created_by', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', json_type, server_default='{}'),
        schema='public' if is_postgresql else None
    )
    
    op.create_index('idx_workflows_created_by', 'workflows', ['created_by'])
    op.create_index('idx_workflows_updated_at', 'workflows', ['updated_at'])
    
    # ============================================================================
    # 2. WORKFLOW_VERSIONS (Immutable Snapshots)
    # ============================================================================
    op.create_table(
        'workflow_versions',
        sa.Column('workflow_version_id', sa.String(255), primary_key=True),
        sa.Column('workflow_id', sa.String(255), sa.ForeignKey('workflows.workflow_id', ondelete='CASCADE'), nullable=False),
        sa.Column('version_number', sa.Integer(), nullable=False),
        sa.Column('ingestion_profile_id', sa.String(255), nullable=False),
        sa.Column('chain_version_id', sa.String(255), sa.ForeignKey('chain_versions.chain_version_id'), nullable=False),
        sa.Column('export_profile_id', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('workflow_id', 'version_number', name='uq_workflow_version'),
        schema='public' if is_postgresql else None
    )
    
    op.create_index('idx_workflow_versions_workflow_id', 'workflow_versions', ['workflow_id'])
    op.create_index('idx_workflow_versions_chain_version_id', 'workflow_versions', ['chain_version_id'])
    
    # ============================================================================
    # 3. WORKFLOW_DOMAINS (Many-to-Many)
    # ============================================================================
    op.create_table(
        'workflow_domains',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('workflow_id', sa.String(255), sa.ForeignKey('workflows.workflow_id', ondelete='CASCADE'), nullable=False),
        sa.Column('domain', sa.String(255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('workflow_id', 'domain', name='uq_workflow_domain'),
        schema='public' if is_postgresql else None
    )
    
    op.create_index('idx_workflow_domains_workflow_id', 'workflow_domains', ['workflow_id'])
    op.create_index('idx_workflow_domains_domain', 'workflow_domains', ['domain'])
    
    # ============================================================================
    # 4. INGESTION_PROFILES
    # ============================================================================
    op.create_table(
        'ingestion_profiles',
        sa.Column('ingestion_profile_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('accepted_input_types', array_type, nullable=False),  # ['PDF', 'DOCX', 'TXT', 'MD', 'CSV']
        sa.Column('mode', sa.String(50), nullable=False),  # 'programmatic' | 'vision'
        sa.Column('vision_prompt', sa.Text(), nullable=True),  # Stored in DB, not file path
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', json_type, server_default='{}'),
        schema='public' if is_postgresql else None
    )
    
    op.create_index('idx_ingestion_profiles_mode', 'ingestion_profiles', ['mode'])
    
    # ============================================================================
    # 5. EXPORT_PROFILES
    # ============================================================================
    op.create_table(
        'export_profiles',
        sa.Column('export_profile_id', sa.String(255), primary_key=True),
        sa.Column('name', sa.String(500), nullable=False),
        sa.Column('format', sa.String(50), nullable=False),  # 'CSV' | 'JSON' | 'MD' | 'DOCX' | 'PDF'
        sa.Column('config_json', json_type, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        schema='public' if is_postgresql else None
    )
    
    op.create_index('idx_export_profiles_format', 'export_profiles', ['format'])
    
    # ============================================================================
    # 6. EXECUTION_TASKS (For CSV row-based execution)
    # ============================================================================
    op.create_table(
        'execution_tasks',
        sa.Column('task_id', sa.String(255), primary_key=True),
        sa.Column('run_id', sa.String(255), sa.ForeignKey('runs.run_id', ondelete='CASCADE'), nullable=False),
        sa.Column('doc_id', sa.String(255), sa.ForeignKey('documents.doc_id', ondelete='CASCADE'), nullable=False),
        sa.Column('row_index', sa.Integer(), nullable=False),
        sa.Column('row_data', json_type, nullable=False),  # Serialized row data
        sa.Column('status', sa.String(50), nullable=False),  # 'QUEUED' | 'RUNNING' | 'SUCCESS' | 'ERROR'
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('run_id', 'doc_id', 'row_index', name='uq_execution_task'),
        schema='public' if is_postgresql else None
    )
    
    op.create_index('idx_execution_tasks_run_id', 'execution_tasks', ['run_id'])
    op.create_index('idx_execution_tasks_doc_id', 'execution_tasks', ['doc_id'])
    op.create_index('idx_execution_tasks_status', 'execution_tasks', ['status'])
    
    # ============================================================================
    # 7. MODIFY EXISTING TABLES
    # ============================================================================
    
    # Add title to chain_steps (nullable initially, will be made NOT NULL in later migration)
    op.add_column('chain_steps', sa.Column('title', sa.String(500), nullable=True))
    
    # Add workflow_version_id to runs (nullable initially)
    op.add_column('runs', sa.Column('workflow_version_id', sa.String(255), sa.ForeignKey('workflow_versions.workflow_version_id'), nullable=True))
    op.create_index('idx_runs_workflow_version_id', 'runs', ['workflow_version_id'])
    
    # Add task_id to step_results (nullable, for CSV task-based execution)
    op.add_column('step_results', sa.Column('task_id', sa.String(255), sa.ForeignKey('execution_tasks.task_id'), nullable=True))
    op.create_index('idx_step_results_task_id', 'step_results', ['task_id'])


def downgrade() -> None:
    """Remove workflow system tables and revert modifications."""
    
    # Drop indexes first
    op.drop_index('idx_step_results_task_id', 'step_results')
    op.drop_index('idx_runs_workflow_version_id', 'runs')
    op.drop_index('idx_execution_tasks_status', 'execution_tasks')
    op.drop_index('idx_execution_tasks_doc_id', 'execution_tasks')
    op.drop_index('idx_execution_tasks_run_id', 'execution_tasks')
    op.drop_index('idx_export_profiles_format', 'export_profiles')
    op.drop_index('idx_ingestion_profiles_mode', 'ingestion_profiles')
    op.drop_index('idx_workflow_domains_domain', 'workflow_domains')
    op.drop_index('idx_workflow_domains_workflow_id', 'workflow_domains')
    op.drop_index('idx_workflow_versions_chain_version_id', 'workflow_versions')
    op.drop_index('idx_workflow_versions_workflow_id', 'workflow_versions')
    op.drop_index('idx_workflows_updated_at', 'workflows')
    op.drop_index('idx_workflows_created_by', 'workflows')
    
    # Drop columns from existing tables
    op.drop_column('step_results', 'task_id')
    op.drop_column('runs', 'workflow_version_id')
    op.drop_column('chain_steps', 'title')
    
    # Drop new tables
    op.drop_table('execution_tasks')
    op.drop_table('export_profiles')
    op.drop_table('ingestion_profiles')
    op.drop_table('workflow_domains')
    op.drop_table('workflow_versions')
    op.drop_table('workflows')

